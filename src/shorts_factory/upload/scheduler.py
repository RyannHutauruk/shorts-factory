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


@dataclass
class ScheduleConfig:
    """In-memory schedule config; see module docstring for the TOML schema."""

    slots: list[str] = field(default_factory=lambda: ["09:30", "20:00"])
    timezone: str = "UTC"
    queue_path: Path = field(default_factory=lambda: Path("~/shorts-factory/queue.txt"))
    out_root: Path = field(default_factory=lambda: Path("~/shorts-factory/out"))
    history_path: Path = field(
        default_factory=lambda: Path("~/.config/shorts-factory/history.json")
    )
    niche: str = "history"
    privacy_status: str = "public"
    upload: bool = True
    auto_refill_topics: bool = True
    refill_count: int = 30
    schedule_publish_offset_min: int = 0

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
            privacy_status=self.privacy_status,
            upload=self.upload,
            auto_refill_topics=self.auto_refill_topics,
            refill_count=self.refill_count,
            schedule_publish_offset_min=self.schedule_publish_offset_min,
        )


def load_config(path: Path) -> ScheduleConfig:
    """Parse a schedule.toml file. Unknown keys are ignored."""
    raw = _toml.loads(path.read_text(encoding="utf-8"))
    cfg = ScheduleConfig(
        slots=list(raw.get("slots", ["09:30", "20:00"])),
        timezone=str(raw.get("timezone", "UTC")),
        queue_path=Path(str(raw.get("queue_path", "~/shorts-factory/queue.txt"))),
        out_root=Path(str(raw.get("out_root", "~/shorts-factory/out"))),
        history_path=Path(str(raw.get("history_path", "~/.config/shorts-factory/history.json"))),
        niche=str(raw.get("niche", "history")),
        privacy_status=str(raw.get("privacy_status", "public")),
        upload=bool(raw.get("upload", True)),
        auto_refill_topics=bool(raw.get("auto_refill_topics", True)),
        refill_count=int(raw.get("refill_count", 30)),
        schedule_publish_offset_min=int(raw.get("schedule_publish_offset_min", 0)),
    )
    return cfg.expanded()


def write_default_config(path: Path) -> None:
    """Create a starter schedule.toml at ``path`` if missing."""
    if path.exists():
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
niche = "history"            # true_crime | history | science | mysteries | weird_facts | biographies | tech_history | space
privacy_status = "public"    # public | unlisted | private
upload = true                # set false to generate without uploading
auto_refill_topics = true    # auto-call Gemini topic discovery when queue empties
refill_count = 30
schedule_publish_offset_min = 0
""",
        encoding="utf-8",
    )


@dataclass
class TickResult:
    topic: str | None
    short_path: Path | None
    upload_url: str | None
    error: str | None


def tick_once(cfg: ScheduleConfig) -> TickResult:
    """Run a single scheduler tick: pop a topic, generate, optionally upload."""
    from ..niche.batch import BatchHistory, pop_next_topic
    from ..niche.pipeline import run_niche_pipeline
    from ..niche.topics import discover_topics, render_topics_text

    cfg = cfg.expanded()
    cfg.queue_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_root.mkdir(parents=True, exist_ok=True)

    history = BatchHistory.load(cfg.history_path)

    if not cfg.queue_path.exists() or _queue_is_empty(cfg.queue_path):
        if not cfg.auto_refill_topics:
            return TickResult(topic=None, short_path=None, upload_url=None, error="queue empty")
        topics = discover_topics(niche=cfg.niche, count=cfg.refill_count, avoid=history.topics)
        cfg.queue_path.write_text(render_topics_text(topics) + "\n", encoding="utf-8")
        print(f"[schedule] refilled queue with {len(topics)} topics")

    topic = pop_next_topic(cfg.queue_path, history)
    if not topic:
        return TickResult(
            topic=None, short_path=None, upload_url=None, error="queue empty after refill"
        )

    sub = cfg.out_root / _slug_dir(topic)
    try:
        result = run_niche_pipeline(topic=topic, niche=cfg.niche, out_dir=sub)
    except Exception as exc:
        return TickResult(topic=topic, short_path=None, upload_url=None, error=f"generate: {exc}")

    history.record(topic, result.script.hook)
    history.save()

    if not cfg.upload:
        return TickResult(topic=topic, short_path=result.short_path, upload_url=None, error=None)

    return _upload_and_record(cfg, result, sub)


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


def run_forever(cfg_path: Path) -> None:
    """Run the scheduler indefinitely (blocking). Ctrl-C to stop."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("apscheduler not installed. Run `uv pip install apscheduler`.") from exc

    cfg = load_config(cfg_path)
    sched = BlockingScheduler(timezone=cfg.timezone)

    def _job() -> None:
        result = tick_once(cfg)
        if result.error:
            print(f"[schedule] error: {result.error}")
        else:
            url = result.upload_url or "(generation-only)"
            print(f"[schedule] ok: {result.topic} -> {url}")

    for slot in cfg.slots:
        try:
            hh, mm = slot.split(":", 1)
            sched.add_job(
                _job,
                CronTrigger(hour=int(hh), minute=int(mm)),
                id=f"slot-{slot}",
                replace_existing=True,
            )
        except ValueError:
            print(f"[schedule] invalid slot {slot!r}, expected HH:MM", file=sys.stderr)
    print(
        f"[schedule] running. cadence={len(cfg.slots)}/day, "
        f"slots={cfg.slots} tz={cfg.timezone}, niche={cfg.niche}, "
        f"upload={'on' if cfg.upload else 'off'}"
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
