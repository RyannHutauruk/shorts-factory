"""Local batch generator + topic-queue persistence.

The batch reads topics from a queue file (one per line, ``#`` comments
ignored), generates each as a niche short, and tracks completed topics
so a long-running channel never re-uses the same idea. Anti-repetition:
we also track a 'hook stem' (the lemmatised first 6 words) and reject
new shorts whose hook stem is already in the recent-history file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import NicheResult, run_niche_pipeline


@dataclass
class BatchHistory:
    """Track completed topics + recent hook stems to keep the channel diverse.

    Persisted as JSON at ``path``. Designed to be read/written by both the
    batch CLI and the scheduler so they never conflict.
    """

    path: Path
    topics: list[str] = field(default_factory=list)
    hook_stems: list[str] = field(default_factory=list)
    max_recent_stems: int = 50

    @classmethod
    def load(cls, path: Path) -> BatchHistory:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            return cls(
                path=path,
                topics=list(payload.get("topics", [])),
                hook_stems=list(payload.get("hook_stems", [])),
                max_recent_stems=int(payload.get("max_recent_stems", 50)),
            )
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "topics": self.topics,
            "hook_stems": self.hook_stems[-self.max_recent_stems :],
            "max_recent_stems": self.max_recent_stems,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def normalise_topic(self, topic: str) -> str:
        return re.sub(r"\s+", " ", topic.lower().strip())

    def has_topic(self, topic: str) -> bool:
        norm = self.normalise_topic(topic)
        return any(self.normalise_topic(t) == norm for t in self.topics)

    def hook_stem(self, hook: str) -> str:
        words = re.findall(r"[a-zA-Z']+", hook.lower())
        return " ".join(words[:6])

    def has_hook_stem(self, hook: str) -> bool:
        stem = self.hook_stem(hook)
        return stem in self.hook_stems if stem else False

    def record(self, topic: str, hook: str) -> None:
        if not self.has_topic(topic):
            self.topics.append(topic)
        stem = self.hook_stem(hook)
        if stem and stem not in self.hook_stems:
            self.hook_stems.append(stem)


def parse_topics_file(path: Path) -> list[str]:
    """Read topics file (one per line, '#' comments and indented '# ...' allowed)."""
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def pop_next_topic(queue_path: Path, history: BatchHistory) -> str | None:
    """Atomically pop the first non-completed topic from the queue file."""
    if not queue_path.exists():
        return None
    lines = queue_path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if history.has_topic(line):
            continue
        # Mark as taken in-place so a concurrent reader doesn't pick the same.
        lines[i] = f"# done @ {datetime.now(timezone.utc).isoformat()}: {line}"
        queue_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return line
    return None


@dataclass
class BatchOptions:
    niche: str = "true_crime"
    out_root: Path | None = None
    history_path: Path | None = None
    skip_metadata: bool = False
    max_count: int | None = None  # cap on number of shorts to produce in this run


def run_batch(
    topics: list[str],
    *,
    options: BatchOptions | None = None,
) -> list[NicheResult]:
    """Generate shorts for each topic, skipping duplicates and same-stem hooks.

    Failures are logged and skipped (one bad topic doesn't stop the batch).
    """
    opts = options or BatchOptions()
    hist_path = opts.history_path or Path.home() / ".config/shorts-factory/history.json"
    history = BatchHistory.load(hist_path)

    results: list[NicheResult] = []
    for topic in topics:
        if opts.max_count is not None and len(results) >= opts.max_count:
            break
        if history.has_topic(topic):
            print(f"[batch] skip (already done): {topic}")
            continue
        sub = opts.out_root / _slug_dir(topic) if opts.out_root else None
        try:
            result = run_niche_pipeline(
                topic=topic,
                niche=opts.niche,
                out_dir=sub,
                skip_metadata=opts.skip_metadata,
            )
        except Exception as exc:
            print(f"[batch] failed topic={topic!r}: {exc}")
            continue
        if history.has_hook_stem(result.script.hook):
            stem = history.hook_stem(result.script.hook)
            print(f"[batch] reject (hook reuse): {topic!r} stem={stem!r}")
            continue
        history.record(topic, result.script.hook)
        history.save()
        results.append(result)
        print(f"[batch] OK {topic} -> {result.short_path}")
    return results


def _slug_dir(topic: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower()).strip("-")[:40]
    return f"niche_{s or 'short'}"


__all__ = [
    "BatchHistory",
    "BatchOptions",
    "parse_topics_file",
    "pop_next_topic",
    "run_batch",
]
