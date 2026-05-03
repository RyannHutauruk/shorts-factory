"""Step 2 - download a movie file from archive.org via yt-dlp."""

from __future__ import annotations

from pathlib import Path

from yt_dlp import YoutubeDL

from .config import PATHS
from .discovery import DETAIL_URL


def download(identifier: str, *, dest: Path | None = None, format_str: str | None = None) -> Path:
    """Download the best-available video for `identifier` and return the local path.

    yt-dlp's `archive.org` extractor handles the per-file selection from the
    item's manifest. We prefer mp4 with the largest height to keep the rest of
    the pipeline simple.
    """
    PATHS.ensure()
    out_dir = dest or PATHS.downloads
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = format_str or "bv*[ext=mp4]+ba/b[ext=mp4]/bv*+ba/b"
    outtmpl = str(out_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "noprogress": False,
        "quiet": False,
        "no_warnings": False,
        "concurrent_fragment_downloads": 4,
        "retries": 5,
    }

    url = DETAIL_URL.format(identifier=identifier)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # archive.org playlists return a playlist info dict; pick the first entry
        if info.get("_type") == "playlist":
            entries = [e for e in info.get("entries", []) if e]
            if not entries:
                raise RuntimeError(f"No downloadable entries found for {identifier}")
            info = entries[0]
        local = Path(ydl.prepare_filename(info))
        if not local.exists():
            # yt-dlp may have remuxed to a different ext
            stem = local.with_suffix("")
            for ext in (".mp4", ".mkv", ".webm"):
                cand = stem.with_suffix(ext)
                if cand.exists():
                    return cand
            raise FileNotFoundError(f"Expected downloaded file at {local} but it does not exist")
        return local
