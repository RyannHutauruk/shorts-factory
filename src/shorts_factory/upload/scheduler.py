"""Local scheduler: generate + upload N shorts/day on your own machine.

Reads ``~/.config/shorts-factory/schedule.toml`` and runs the
generate-then-upload cycle at the configured slot times. Designed to be
launched once (``shorts-factory schedule run``) and left running, e.g.
inside ``tmux``, ``screen``, or as a systemd user service.

Schedule file example::

    cadence_per_day = 2
    slots = ["09:30", "20:00"]
    timezone = "America/Los_Angeles"
    queue_path = "~/shorts-factory/queue.txt"
    out_root = "~/shorts-factory/out"
    history_path = "~/.config/shorts-factory/history.json"
    niche = "history"
    privacy_status = "public"   # public | unlisted | private
    upload = true
    schedule_publish_offset_min = 0  # 0 = upload immediately public; >0 = upload private + publishAt

The scheduler:
1. At each slot, pops the next un-completed topic from ``queue_path``.
2. Generates the short via ``run_niche_pipeline``.
3. Uploads via the local OAuth-cached YouTubeUploader (only if ``upload = true``).
4. Records success in ``history.json``.
5. If the queue is empty, optionally calls ``discover_topics`` to refill
   it (default 30 fresh topics).
"""

from __future__ import annotations

import os
import random
import re
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:  # py3.11+ ships tomllib; py3.10 needs tomli (already a transitive dep)
    import tomllib as _toml
except ImportError:  # pragma: no cover - py<3.11
    import tomli as _toml


ALL_NICHES: list[str] = [
    "true_crime",
    "history",
    "science",
    "mysteries",
    "weird_facts",
    "biographies",
    "tech_history",
    "space",
]


@dataclass
class ScheduleConfig:
    """In-memory schedule config; see module docstring for the TOML schema.

    ``niche`` may be a single string or a list of strings. When it's a list
    (or the special value ``"all"``) each tick picks a niche at random.
    The previous niche is tracked in-memory and avoided on the next pick if
    there are 2+ niches in the rotation.
    """

    slots: list[str] = field(default_factory=lambda: ["09:30", "20:00"])
    timezone: str = "UTC"
    queue_path: Path = field(default_factory=lambda: Path("~/shorts-factory/queue.txt"))
    out_root: Path = field(default_factory=lambda: Path("~/shorts-factory/out"))
    history_path: Path = field(
        default_factory=lambda: Path("~/.config/shorts-factory/history.json")
    )
    niche: str | list[str] = "history"
    audience: str = "US"
    privacy_status: str = "public"
    upload: bool = True
    auto_refill_topics: bool = True
    refill_count: int = 30
    schedule_publish_offset_min: int = 0
    last_niche: str | None = None  # in-memory only; reset per process

    # Long-form documentary scheduler (separate cadence from shorts).
    # Empty list = long-form disabled (default).
    longform_slots: list[str] = field(default_factory=list)
    # Cron day-of-week filter for long-form, e.g. "mon,wed,fri".
    # Empty = every day.
    longform_days: str = ""
    longform_duration_min: float = 10.0
    longform_niche: str | list[str] = field(default_factory=list)
    longform_voices: list[Path] = field(default_factory=list)
    longform_music: Path | None = None
    longform_privacy_status: str = "private"
    longform_upload: bool = True
    last_longform_niche: str | None = None

    @staticmethod
    def _expand(p: Path) -> Path:
        return Path(os.path.expanduser(str(p))).resolve()

    def expanded(self) -> ScheduleConfig:
        return ScheduleConfig(
            slots=list(self.slots),
            timezone=self.timezone,
            queue_path=self._expand(self.queue_path),
            out_root=self._expand(self.out_root),
            history_path=self._expand(self.history_path),
            niche=self.niche,
            audience=self.audience,
            privacy_status=self.privacy_status,
            upload=self.upload,
            auto_refill_topics=self.auto_refill_topics,
            refill_count=self.refill_count,
            schedule_publish_offset_min=self.schedule_publish_offset_min,
            last_niche=self.last_niche,
            longform_slots=list(self.longform_slots),
            longform_days=self.longform_days,
            longform_duration_min=self.longform_duration_min,
            longform_niche=self.longform_niche,
            longform_voices=[self._expand(v) for v in self.longform_voices],
            longform_music=self._expand(self.longform_music) if self.longform_music else None,
            longform_privacy_status=self.longform_privacy_status,
            longform_upload=self.longform_upload,
            last_longform_niche=self.last_longform_niche,
        )

    def niche_list(self) -> list[str]:
        """Resolve ``niche`` (string | list | 'all') to a list of valid niches."""
        if isinstance(self.niche, list):
            niches = [n for n in self.niche if n in ALL_NICHES]
            if not niches:
                raise ValueError(f"niche list contains no recognised entries: {self.niche!r}")
            return niches
        if self.niche == "all":
            return list(ALL_NICHES)
        if self.niche in ALL_NICHES:
            return [self.niche]
        raise ValueError(f"unknown niche {self.niche!r}; choose from {ALL_NICHES} or pass a list")

    def pick_niche(self, *, rng: random.Random | None = None) -> str:
        """Pick the niche for the next tick. Avoids back-to-back repeats when
        the rotation has 2+ niches.
        """
        niches = self.niche_list()
        if len(niches) == 1:
            return niches[0]
        candidates = [n for n in niches if n != self.last_niche] or niches
        if rng is None:
            return random.choice(candidates)
        return rng.choice(candidates)

    def queue_path_for(self, niche: str) -> Path:
        """Per-niche queue file. For multi-niche rotations we treat
        ``queue_path`` as a template: ``queue.txt`` -> ``queue_<niche>.txt``
        in the same directory.
        """
        if isinstance(self.niche, str) and self.niche != "all":
            return self.queue_path
        suffix = self.queue_path.suffix or ".txt"
        stem = self.queue_path.stem or "queue"
        return self.queue_path.with_name(f"{stem}_{niche}{suffix}")

    def longform_niche_list(self) -> list[str]:
        """Resolve ``longform_niche`` to a niche list. Falls back to ``niche``
        when ``longform_niche`` is empty so users only have to set it once
        if they want the same rotation for both formats.
        """
        if isinstance(self.longform_niche, list) and self.longform_niche:
            niches = [n for n in self.longform_niche if n in ALL_NICHES]
            if not niches:
                raise ValueError(
                    f"longform_niche list contains no recognised entries: {self.longform_niche!r}"
                )
            return niches
        if isinstance(self.longform_niche, str):
            if self.longform_niche == "all":
                return list(ALL_NICHES)
            if self.longform_niche in ALL_NICHES:
                return [self.longform_niche]
        # Fall through to the shorts niche list.
        return self.niche_list()

    def pick_longform_niche(self, *, rng: random.Random | None = None) -> str:
        niches = self.longform_niche_list()
        if len(niches) == 1:
            return niches[0]
        candidates = [n for n in niches if n != self.last_longform_niche] or niches
        if rng is None:
            return random.choice(candidates)
        return rng.choice(candidates)

    def longform_queue_path_for(self, niche: str) -> Path:
        """Long-form queue files are kept separate so we don't burn through
        shorts topics with documentaries. ``queue_history.txt`` ->
        ``queue_longform_history.txt``.
        """
        suffix = self.queue_path.suffix or ".txt"
        stem = self.queue_path.stem or "queue"
        return self.queue_path.with_name(f"{stem}_longform_{niche}{suffix}")


def load_config(path: Path) -> ScheduleConfig:
    """Parse a schedule.toml file. Unknown keys are ignored.

    ``niche`` may be a string, a list of strings, or the special value
    ``"all"`` (rotate through all 8 built-in niches).
    """
    raw = _toml.loads(path.read_text(encoding="utf-8"))
    raw_niche = raw.get("niche", "history")
    niche: str | list[str]
    if isinstance(raw_niche, list):
        niche = [str(n) for n in raw_niche]
    else:
        niche = str(raw_niche)
    raw_lf_niche = raw.get("longform_niche", [])
    longform_niche: str | list[str]
    if isinstance(raw_lf_niche, list):
        longform_niche = [str(n) for n in raw_lf_niche]
    else:
        longform_niche = str(raw_lf_niche)

    cfg = ScheduleConfig(
        slots=list(raw.get("slots", ["09:30", "20:00"])),
        timezone=str(raw.get("timezone", "UTC")),
        queue_path=Path(str(raw.get("queue_path", "~/shorts-factory/queue.txt"))),
        out_root=Path(str(raw.get("out_root", "~/shorts-factory/out"))),
        history_path=Path(str(raw.get("history_path", "~/.config/shorts-factory/history.json"))),
        niche=niche,
        audience=str(raw.get("audience", "US")),
        privacy_status=str(raw.get("privacy_status", "public")),
        upload=bool(raw.get("upload", True)),
        auto_refill_topics=bool(raw.get("auto_refill_topics", True)),
        refill_count=int(raw.get("refill_count", 30)),
        schedule_publish_offset_min=int(raw.get("schedule_publish_offset_min", 0)),
        longform_slots=list(raw.get("longform_slots", [])),
        longform_days=str(raw.get("longform_days", "")),
        longform_duration_min=float(raw.get("longform_duration_min", 10.0)),
        longform_niche=longform_niche,
        longform_voices=[Path(str(v)) for v in raw.get("longform_voices", [])],
        longform_music=Path(str(raw["longform_music"])) if raw.get("longform_music") else None,
        longform_privacy_status=str(raw.get("longform_privacy_status", "private")),
        longform_upload=bool(raw.get("longform_upload", True)),
    )
    return cfg.expanded()


def write_default_config(path: Path, *, force: bool = False) -> None:
    """Create a starter schedule.toml at ``path``. ``force`` overwrites."""
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# shorts-factory local scheduler config.
# Edit the values, then run:  shorts-factory schedule run
slots = ["09:30", "20:00"]   # 2 uploads/day, in your local timezone
timezone = "UTC"             # set to your local TZ, e.g. "America/Los_Angeles"
queue_path = "~/shorts-factory/queue.txt"
out_root = "~/shorts-factory/out"
history_path = "~/.config/shorts-factory/history.json"

# Niche selection. Either:
#   niche = "history"                           # single niche
#   niche = "all"                               # rotate through every built-in niche
#   niche = ["history", "science", "mysteries"] # rotate through your hand-picked subset
niche = ["true_crime", "history", "science", "mysteries", "weird_facts", "biographies", "tech_history", "space"]

audience = "US"              # US | UK | global | general - biases topic discovery
privacy_status = "public"    # public | unlisted | private
upload = true                # set false to generate without uploading
auto_refill_topics = true    # auto-call Gemini topic discovery when a niche queue empties
refill_count = 30
schedule_publish_offset_min = 0

# ---------- Long-form documentaries (10-min 16:9 videos) ----------
# Long-form runs on a SEPARATE cadence (much heavier per render).
# Leave longform_slots = [] to disable. Recommended cadence: 2-3 per WEEK.
longform_slots = []                    # e.g. ["08:00"] - one render per active day
longform_days  = ""                    # e.g. "mon,wed,fri" - blank means daily
longform_duration_min = 10.0           # 5-25 minutes typical
longform_niche = []                    # empty = same niche rotation as shorts above
longform_voices = []                   # paths to extra Piper .onnx voices, e.g. ["work/voices/en_US-amy-medium.onnx"]
longform_music = ""                    # optional path to a ducked music bed
longform_privacy_status = "private"    # ALWAYS start private - long-form quality matters more
longform_upload = true
""",
        encoding="utf-8",
    )


@dataclass
class TickResult:
    topic: str | None
    short_path: Path | None
    upload_url: str | None
    error: str | None
    niche: str | None = None


def tick_once(cfg: ScheduleConfig) -> TickResult:
    """Run a single scheduler tick: pick a niche, pop a topic, generate, optionally upload."""
    from ..niche.batch import BatchHistory, pop_next_topic
    from ..niche.pipeline import run_niche_pipeline
    from ..niche.topics import discover_topics, render_topics_text

    cfg = cfg.expanded()
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    niche = cfg.pick_niche()
    cfg.last_niche = niche
    queue_path = cfg.queue_path_for(niche)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[schedule] tick niche={niche} queue={queue_path}")

    history = BatchHistory.load(cfg.history_path)

    if not queue_path.exists() or _queue_is_empty(queue_path):
        if not cfg.auto_refill_topics:
            return TickResult(
                topic=None, short_path=None, upload_url=None, error="queue empty", niche=niche
            )
        topics = discover_topics(
            niche=niche,
            count=cfg.refill_count,
            avoid=history.topics,
            audience=cfg.audience,
        )
        queue_path.write_text(render_topics_text(topics) + "\n", encoding="utf-8")
        print(f"[schedule] refilled {niche} queue with {len(topics)} topics")

    topic = pop_next_topic(queue_path, history)
    if not topic:
        return TickResult(
            topic=None,
            short_path=None,
            upload_url=None,
            error="queue empty after refill",
            niche=niche,
        )

    sub = cfg.out_root / _slug_dir(topic)
    try:
        result = run_niche_pipeline(topic=topic, niche=niche, out_dir=sub)
    except Exception as exc:
        return TickResult(
            topic=topic,
            short_path=None,
            upload_url=None,
            error=f"generate: {exc}",
            niche=niche,
        )

    history.record(topic, result.script.hook)
    history.save()

    if not cfg.upload:
        return TickResult(
            topic=topic,
            short_path=result.short_path,
            upload_url=None,
            error=None,
            niche=niche,
        )

    tick = _upload_and_record(cfg, result, sub)
    tick.niche = niche
    return tick


def tick_once_longform(cfg: ScheduleConfig) -> TickResult:
    """Run a single long-form scheduler tick: pick a niche, pop a topic,
    generate the documentary, optionally upload."""
    from ..niche.batch import BatchHistory, pop_next_topic
    from ..niche.longform_pipeline import run_longform_pipeline
    from ..niche.topics import discover_topics, render_topics_text

    cfg = cfg.expanded()
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    niche = cfg.pick_longform_niche()
    cfg.last_longform_niche = niche
    queue_path = cfg.longform_queue_path_for(niche)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[schedule] longform tick niche={niche} queue={queue_path}")

    history = BatchHistory.load(cfg.history_path)

    if not queue_path.exists() or _queue_is_empty(queue_path):
        if not cfg.auto_refill_topics:
            return TickResult(
                topic=None,
                short_path=None,
                upload_url=None,
                error="longform queue empty",
                niche=niche,
            )
        topics = discover_topics(
            niche=niche,
            count=max(10, cfg.refill_count // 3),  # long-form burns topics slower
            avoid=history.topics,
            audience=cfg.audience,
        )
        queue_path.write_text(render_topics_text(topics) + "\n", encoding="utf-8")
        print(f"[schedule] refilled longform {niche} queue with {len(topics)} topics")

    topic = pop_next_topic(queue_path, history)
    if not topic:
        return TickResult(
            topic=None,
            short_path=None,
            upload_url=None,
            error="longform queue empty after refill",
            niche=niche,
        )

    sub = cfg.out_root / f"longform_{_slug_dir(topic).removeprefix('niche_')}"
    try:
        result = run_longform_pipeline(
            topic=topic,
            niche=niche,
            duration_min=cfg.longform_duration_min,
            audience=cfg.audience,
            voices=cfg.longform_voices or None,
            music_path=cfg.longform_music,
            out_dir=sub,
        )
    except Exception as exc:  # noqa: BLE001
        return TickResult(
            topic=topic,
            short_path=None,
            upload_url=None,
            error=f"longform-generate: {exc}",
            niche=niche,
        )

    history.record(topic, result.script.cold_open[:120])
    history.save()

    if not cfg.longform_upload:
        return TickResult(
            topic=topic,
            short_path=result.video_path,
            upload_url=None,
            error=None,
            niche=niche,
        )

    return _upload_longform_and_record(cfg, result, sub, niche=niche)


def _upload_longform_and_record(
    cfg: ScheduleConfig, result: Any, sub: Path, *, niche: str
) -> TickResult:
    from .youtube import UploadOptions, YouTubeAuthError, YouTubeUploader

    opts = UploadOptions(
        title=result.metadata.title,
        description=result.metadata.full_description,
        tags=[t.lstrip("#") for t in result.metadata.hashtags],
        privacy_status=cfg.longform_privacy_status,
        contains_synthetic_media=True,
    )
    try:
        uploader = YouTubeUploader()
        ur = uploader.upload(result.video_path, opts)
    except YouTubeAuthError as exc:
        return TickResult(
            topic=result.script.topic,
            short_path=result.video_path,
            upload_url=None,
            error=f"longform-upload-auth: {exc}",
            niche=niche,
        )
    except Exception as exc:  # noqa: BLE001
        return TickResult(
            topic=result.script.topic,
            short_path=result.video_path,
            upload_url=None,
            error=f"longform-upload: {exc}",
            niche=niche,
        )

    (sub / "UPLOAD.txt").write_text(
        f"video_id: {ur.video_id}\nurl: {ur.url}\nuploaded_at: {datetime.now().isoformat()}\n",
        encoding="utf-8",
    )
    return TickResult(
        topic=result.script.topic,
        short_path=result.video_path,
        upload_url=ur.url,
        error=None,
        niche=niche,
    )


def _upload_and_record(cfg: ScheduleConfig, result: Any, sub: Path) -> TickResult:
    from .youtube import UploadOptions, YouTubeAuthError, YouTubeUploader

    publish_at = None
    privacy = cfg.privacy_status
    if cfg.schedule_publish_offset_min > 0:
        from datetime import timedelta, timezone

        publish_at = (
            datetime.now(timezone.utc) + timedelta(minutes=cfg.schedule_publish_offset_min)
        ).isoformat(timespec="seconds")
        privacy = "private"

    opts = UploadOptions(
        title=result.metadata.title,
        description=result.metadata.full_description,
        tags=[t.lstrip("#") for t in result.metadata.hashtags],
        privacy_status=privacy,
        publish_at=publish_at,
        contains_synthetic_media=True,
    )
    try:
        uploader = YouTubeUploader()
        ur = uploader.upload(result.short_path, opts)
    except YouTubeAuthError as exc:
        return TickResult(
            topic=result.script.topic,
            short_path=result.short_path,
            upload_url=None,
            error=f"upload-auth: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return TickResult(
            topic=result.script.topic,
            short_path=result.short_path,
            upload_url=None,
            error=f"upload: {exc}",
        )

    (sub / "UPLOAD.txt").write_text(
        f"video_id: {ur.video_id}\nurl: {ur.url}\nuploaded_at: {datetime.now().isoformat()}\n",
        encoding="utf-8",
    )
    return TickResult(
        topic=result.script.topic, short_path=result.short_path, upload_url=ur.url, error=None
    )


def _add_cron_slots(
    sched: Any,
    slots: list[str],
    job_callable: Any,
    *,
    label: str,
    day_of_week: str = "",
) -> None:
    from apscheduler.triggers.cron import CronTrigger

    for slot in slots:
        try:
            hh, mm = slot.split(":", 1)
            kwargs: dict[str, Any] = {"hour": int(hh), "minute": int(mm)}
            if day_of_week:
                kwargs["day_of_week"] = day_of_week
            sched.add_job(
                job_callable,
                CronTrigger(**kwargs),
                id=f"{label}-slot-{slot}",
                replace_existing=True,
            )
        except ValueError:
            print(f"[schedule] invalid {label} slot {slot!r}, expected HH:MM", file=sys.stderr)


def run_forever(cfg_path: Path) -> None:
    """Run the scheduler indefinitely (blocking). Ctrl-C to stop."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("apscheduler not installed. Run `uv pip install apscheduler`.") from exc

    cfg = load_config(cfg_path)
    sched = BlockingScheduler(timezone=cfg.timezone)

    def _shorts_job() -> None:
        result = tick_once(cfg)
        if result.error:
            print(f"[schedule] shorts error: {result.error}")
        else:
            url = result.upload_url or "(generation-only)"
            print(f"[schedule] shorts ok: {result.topic} -> {url}")

    def _longform_job() -> None:
        result = tick_once_longform(cfg)
        if result.error:
            print(f"[schedule] longform error: {result.error}")
        else:
            url = result.upload_url or "(generation-only)"
            print(f"[schedule] longform ok: {result.topic} -> {url}")

    _add_cron_slots(sched, cfg.slots, _shorts_job, label="shorts")
    _add_cron_slots(
        sched,
        cfg.longform_slots,
        _longform_job,
        label="longform",
        day_of_week=cfg.longform_days,
    )

    print(
        f"[schedule] running. shorts={len(cfg.slots)}/day {cfg.slots} "
        f"longform={len(cfg.longform_slots)}/run {cfg.longform_slots} "
        f"days={cfg.longform_days or 'every'} "
        f"tz={cfg.timezone} upload={'on' if cfg.upload else 'off'}"
    )
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("[schedule] stopping")


def _queue_is_empty(path: Path) -> bool:
    if not path.exists():
        return True
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return False
    return True


def _slug_dir(topic: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower()).strip("-")[:40]
    return f"niche_{s or 'short'}"


def systemd_unit(cfg_path: Path, *, exec_path: str | None = None) -> str:
    """Return a sample systemd --user unit file for one-shot install."""
    exe = exec_path or shlex.quote(sys.executable)
    cfg = shlex.quote(str(cfg_path.resolve()))
    return f"""[Unit]
Description=shorts-factory local scheduler
After=network-online.target

[Service]
ExecStart={exe} -m shorts_factory schedule run --config {cfg}
Restart=on-failure
Environment=GEMINI_API_KEY=replace-me

[Install]
WantedBy=default.target
"""


__all__ = [
    "ScheduleConfig",
    "TickResult",
    "load_config",
    "run_forever",
    "systemd_unit",
    "tick_once",
    "tick_once_longform",
    "write_default_config",
]


def _self_test() -> None:  # pragma: no cover - dev helper
    """Smoke test: print the resolved config from the default path."""
    cfg_path = Path("~/.config/shorts-factory/schedule.toml").expanduser()
    if not cfg_path.exists():
        write_default_config(cfg_path)
        print(f"wrote default config -> {cfg_path}")
    cfg = load_config(cfg_path)
    print(cfg)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
