"""Fetch B-roll images from Wikimedia Commons (free, no API key, mostly CC-BY-SA).

Each call to :func:`fetch_broll` returns up to ``max_results`` ``BrollAsset``
records (image downloaded to disk, license + creator captured for the
attribution file). When Wikimedia returns no images for a query, the caller
is expected to fall back to a generic stock asset (handled at the pipeline
level, not here).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

WIKI_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_HEADERS = {
    "User-Agent": (
        "shorts-factory/0.1 (https://github.com/RyannHutauruk/shorts-factory; "
        "rich.panda.real@gmail.com) python-requests/2"
    )
}

# Skip these license tags - non-commercial-only or unsuitable for monetized shorts.
NON_COMMERCIAL_HINTS = ("noncommercial", "non-commercial", "cc-by-nc", "cc-by-nd")

# Skip these in titles/categories - they're typically not useful B-roll
# (book covers, library catalog images, charts, sheet music, postage stamps,
# building floorplans, unrelated maps, etc.).
JUNK_TITLE_HINTS = (
    "book cover",
    "title page",
    "frontispiece",
    "library catalog",
    "library_of_congress",
    "stamp",
    "postage",
    "sheet_music",
    "sheet music",
    "musical_score",
    "musical score",
    "pamphlet",
    "manuscript",
    "page_from",
    "logo",
    "coat_of_arms",
    "coat of arms",
    "diagram",
    "blueprint",
    "floor_plan",
    "floor plan",
    "newspaper_clipping",
    "newspaper clipping",
    "page_001",
    "_page_",
    "letterhead",
    "ticket_stub",
    ".djvu",
    ".pdf",
    "djvu",
    "_pdf_",
    "album_de_luxe",
    "the_war_illustrated",
    "war_illustrated_album",
    "card_index",
    "town_crier",
    "_index_",
    "_index.",
    "_directory_",
    "directory.djvu",
    "encyclop",  # encyclopedia/encyclopaedia
    "scrapbook",
    "annual_report",
    "art_museum",
    "card_catalog",
    "book_images",  # 'Internet Archive Book Images' uploads
    "internet_archive_book",
    "great_settlement",  # generic 1918 book frequently surfaced
)


@dataclass(frozen=True)
class BrollAsset:
    """A single B-roll image with the metadata we need for legal reuse."""

    query: str
    title: str
    local_path: Path
    width: int
    height: int
    mime: str
    license_short: str  # e.g. "CC BY-SA 4.0", "Public domain"
    creator: str
    page_url: str  # Wikimedia file page (the canonical credit URL)

    @property
    def attribution_line(self) -> str:
        """Single ready-to-paste credit line."""
        cred = self.creator or "Wikimedia Commons"
        return f'"{self.title}" by {cred} ({self.license_short}): {self.page_url}'


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _safe_get(d: dict[str, Any], *path: str, default: str = "") -> str:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, {})
    if isinstance(cur, dict):
        return str(cur.get("value", default))
    return str(cur) if cur else default


def _is_acceptable_license(extmetadata: dict[str, Any]) -> bool:
    short = _safe_get(extmetadata, "LicenseShortName").lower()
    licence = _safe_get(extmetadata, "License").lower()
    if not short and not licence:
        # Some PD-old entries have UsageTerms but no LicenseShortName.
        usage = _safe_get(extmetadata, "UsageTerms").lower()
        return "public domain" in usage or "free" in usage
    blob = short + " " + licence
    if any(tag in blob for tag in NON_COMMERCIAL_HINTS):
        return False
    return True


_BAD_IMAGE_MIMES = {
    "image/svg+xml",
    "image/vnd.djvu",
    "image/x-djvu",
    "image/tiff",  # often huge multi-page documents
}


def _is_image_mime(m: str) -> bool:
    return m.startswith("image/") and m not in _BAD_IMAGE_MIMES


def _looks_like_junk(title: str) -> bool:
    """Skip book covers, library catalog scans, sheet music, etc."""
    haystack = title.lower().replace(" ", "_")
    return any(hint.replace(" ", "_") in haystack for hint in JUNK_TITLE_HINTS)


def _normalise_to_png(src: Path) -> Path:
    """Convert webp / unusual formats to PNG via ffmpeg so ffmpeg's image2
    demuxer accepts ``-loop 1`` reliably. Returns the path to the canonical
    file (.jpg or .png)."""
    if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return src
    if shutil.which("ffmpeg") is None:
        return src  # best-effort; will probably fail at assemble step
    out = src.with_suffix(".png")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-frames:v",
            "1",
            str(out),
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0 and out.exists():
        try:
            src.unlink()
        except OSError:
            pass
        return out
    return src


def _page_to_asset(
    page: dict[str, Any],
    *,
    query: str,
    dest_dir: Path,
    min_dimension: int,
    timeout: float,
) -> BrollAsset | None:
    """Validate, download, and convert a single Wikimedia search hit. Returns
    None when the hit fails any filter (mime/license/dimension/junk title)."""
    info_list = page.get("imageinfo") or []
    if not info_list:
        return None
    info = info_list[0]
    mime = str(info.get("mime", ""))
    if not _is_image_mime(mime):
        return None
    meta = info.get("extmetadata", {}) or {}
    if not _is_acceptable_license(meta):
        return None
    width = int(info.get("thumbwidth") or info.get("width") or 0)
    height = int(info.get("thumbheight") or info.get("height") or 0)
    if min(width, height) < min_dimension:
        return None
    thumb_url = info.get("thumburl") or info.get("url")
    if not thumb_url:
        return None
    title = str(page.get("title", "")).removeprefix("File:")
    if _looks_like_junk(title):
        return None

    suffix = ".jpg" if "jpeg" in mime else ".png" if "png" in mime else ".webp"
    digest = hashlib.sha1(thumb_url.encode("utf-8")).hexdigest()[:10]
    local_path = dest_dir / f"{digest}{suffix}"
    if not local_path.exists():
        r = requests.get(thumb_url, headers=DEFAULT_HEADERS, timeout=timeout)
        r.raise_for_status()
        local_path.write_bytes(r.content)
    local_path = _normalise_to_png(local_path)

    page_url = info.get("descriptionurl") or info.get("descriptionshorturl") or ""
    creator = _strip_html(_safe_get(meta, "Artist"))
    license_short = _safe_get(meta, "LicenseShortName") or _safe_get(meta, "License")
    return BrollAsset(
        query=query,
        title=title,
        local_path=local_path,
        width=width,
        height=height,
        mime=mime,
        license_short=license_short or "unspecified",
        creator=creator or "Unknown",
        page_url=str(page_url),
    )


def fetch_broll(
    query: str,
    *,
    dest_dir: Path,
    max_results: int = 3,
    min_dimension: int = 480,
    timeout: float = 30.0,
) -> list[BrollAsset]:
    """Search Wikimedia Commons for images matching ``query`` and download them.

    The result is filtered to commercially-reusable licenses, real raster
    images (no SVG / no PDF), and a minimum width/height.
    """
    if not query.strip():
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, str | int] = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,  # File:
        "gsrlimit": max(10, max_results * 4),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1920,
        "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    pages = ((resp.json() or {}).get("query") or {}).get("pages") or {}

    out: list[BrollAsset] = []
    for _, page in sorted(pages.items(), key=lambda kv: kv[1].get("index", 999)):
        if len(out) >= max_results:
            break
        asset = _page_to_asset(
            page,
            query=query,
            dest_dir=dest_dir,
            min_dimension=min_dimension,
            timeout=timeout,
        )
        if asset is not None:
            out.append(asset)
    return out
