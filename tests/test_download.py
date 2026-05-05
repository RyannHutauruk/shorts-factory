"""Unit tests for the archive.org file picker (download.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from shorts_factory.download import ArchiveFile, list_files, pick_best


def _f(name: str, size_mb: int, fmt: str = "h.264") -> ArchiveFile:
    ext = "." + name.rsplit(".", 1)[-1].lower()
    return ArchiveFile(name=name, size=size_mb * 1024 * 1024, format=fmt, ext=ext)


def test_pick_best_prefers_mp4() -> None:
    files = [
        _f("a_1080p.mkv", 11000, "h.264 MPEG4"),
        _f("a_1080p.mp4", 569, "h.264"),
        _f("a_512kb.mp4", 388, "512Kb MPEG4"),
        _f("a.ogv", 405, "Ogg Video"),
    ]
    chosen = pick_best(files, max_size_mb=4096)
    assert chosen.name == "a_1080p.mp4"


def test_pick_best_skips_oversized_mp4() -> None:
    files = [
        _f("huge.mp4", 8000),
        _f("normal.mp4", 600),
    ]
    chosen = pick_best(files, max_size_mb=4096)
    assert chosen.name == "normal.mp4"


def test_pick_best_falls_back_to_mkv_when_no_mp4_in_band() -> None:
    files = [
        _f("oversized.mp4", 8000),
        _f("ok.mkv", 3000),
        _f("tiny.mp4", 5),
    ]
    chosen = pick_best(files, max_size_mb=4096, min_size_mb=50)
    assert chosen.name == "ok.mkv"


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_list_files_filters_to_video_extensions() -> None:
    payload = {
        "files": [
            {"name": "movie.mp4", "size": "1000000", "format": "h.264"},
            {"name": "movie.mkv", "size": "9999", "format": "h.264 MPEG4"},
            {"name": "thumb.jpg", "size": "1024", "format": "JPEG"},
            {"name": "subs.srt", "size": "500", "format": "SubRip"},
            {"name": "no_ext_file", "size": "10", "format": ""},
        ]
    }
    with patch("shorts_factory.download.requests.get", return_value=_Resp(payload)):
        files = list_files("any-id")
    names = {f.name for f in files}
    assert names == {"movie.mp4", "movie.mkv"}
    sizes = {f.name: f.size for f in files}
    assert sizes["movie.mp4"] == 1_000_000
