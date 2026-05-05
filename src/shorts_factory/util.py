"""Small shared utilities."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def slugify(value: str, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("-", value).strip("-").lower()
    return s[:max_len] or "untitled"


def run(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run that surfaces useful errors."""
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def probe_duration(path: Path) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe could not determine duration of {path}: {out!r}") from exc


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg / ASS."""
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:01d}:{minutes:02d}:{secs:06.3f}"
