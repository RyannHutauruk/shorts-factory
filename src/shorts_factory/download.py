"""Step 2 - download a movie file from archive.org.

We talk to the archive.org metadata API directly so we can pick a *single*
sensibly-sized variant instead of letting yt-dlp download every encoding the
item exposes (some items contain 8+ duplicates totalling tens of GB).

The picker prefers MP4 (h.264) over MKV / MPEG-2 / Ogg because the rest of the
pipeline (PySceneDetect, Whisper, ffmpeg) is happiest with H.264 in MP4. Within
the MP4 candidates it picks the largest file (proxy for highest bitrate /
resolution) under `max_size_mb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import PATHS

METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

_VIDEO_EXTS = (".mp4", ".mkv", ".m2ts", ".mov", ".avi", ".webm", ".mpg", ".mpeg")
_PREFERRED_EXTS = (".mp4", ".mkv")


@dataclass(frozen=True)
class ArchiveFile:
    name: str
    size: int
    format: str
    ext: str


def list_files(identifier: str, *, timeout: float = 30.0) -> list[ArchiveFile]:
    """Return the video files in an archive.org item."""
    url = METADATA_URL.format(identifier=identifier)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    out: list[ArchiveFile] = []
    for f in payload.get("files", []):
        name = str(f.get("name", ""))
        lower = name.lower()
        ext = next((e for e in _VIDEO_EXTS if lower.endswith(e)), "")
        if not ext:
            continue
        size_raw = f.get("size", "0") or "0"
        try:
            size = int(size_raw)
        except (TypeError, ValueError):
            size = 0
        out.append(
            ArchiveFile(
                name=name,
                size=size,
                format=str(f.get("format", "")),
                ext=ext,
            )
        )
    return out


def pick_best(
    files: list[ArchiveFile],
    *,
    max_size_mb: int = 4096,
    min_size_mb: int = 50,
) -> ArchiveFile:
    """Pick the best single file: prefer MP4, then MKV, biggest under cap."""
    if not files:
        raise RuntimeError("No video files in this archive.org item")
    cap = max_size_mb * 1024 * 1024
    floor = min_size_mb * 1024 * 1024

    for ext in _PREFERRED_EXTS:
        candidates = [f for f in files if f.ext == ext and floor <= f.size <= cap]
        if candidates:
            return max(candidates, key=lambda f: f.size)

    # Fall back to anything within the size band
    fallback = [f for f in files if floor <= f.size <= cap]
    if fallback:
        return max(fallback, key=lambda f: f.size)

    # Last resort: smallest above floor (avoid the truly tiny phone variants)
    above_floor = [f for f in files if f.size >= floor]
    if above_floor:
        return min(above_floor, key=lambda f: f.size)

    return max(files, key=lambda f: f.size)


def _stream_to_file(url: str, dest: Path, *, timeout: float = 60.0) -> None:
    """Download `url` to `dest` with progress logging via tqdm."""
    from tqdm import tqdm

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (
            dest.open("wb") as fp,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fp.write(chunk)
                bar.update(len(chunk))


def download(
    identifier: str,
    *,
    dest: Path | None = None,
    max_size_mb: int = 4096,
) -> Path:
    """Download the best variant of `identifier` into `dest` and return its path."""
    PATHS.ensure()
    out_dir = dest or (PATHS.downloads / identifier)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_files(identifier)
    chosen = pick_best(files, max_size_mb=max_size_mb)
    local = out_dir / chosen.name
    if local.exists() and local.stat().st_size == chosen.size and chosen.size > 0:
        return local

    url = DOWNLOAD_URL.format(identifier=identifier, filename=chosen.name)
    _stream_to_file(url, local)
    return local
