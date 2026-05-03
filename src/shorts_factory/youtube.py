"""YouTube source discovery + download for Creative Commons-licensed videos.

Uses the YouTube Data API v3 to find videos where the uploader explicitly
licensed under Creative Commons (CC-BY 3.0). Reuse is permitted with
attribution to the original creator.

Two-step flow because the Data API splits search and metadata:

1. search.list - filter by query + ``videoLicense=creativeCommon`` to get IDs.
2. videos.list - look up duration, view count, license, definition for each ID.

We re-verify ``status.license == "creativeCommon"`` from videos.list since the
search filter is not always perfectly accurate.

Quota cost (default per-day budget = 10,000 units):
    search.list  : 100 units / call (returns up to 50 results)
    videos.list  : 1 unit / call (batched up to 50 IDs)

Set ``YOUTUBE_API_KEY`` in the environment before calling :func:`search_cc`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .config import PATHS

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
WATCH_URL = "https://www.youtube.com/watch?v={id}"

_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<mins>\d+)M)?(?:(?P<secs>\d+)S)?$"
)


@dataclass(frozen=True)
class YouTubeItem:
    video_id: str
    title: str
    channel_id: str
    channel_title: str
    description: str
    duration_seconds: int
    view_count: int
    license: str  # "creativeCommon" or "youtube"
    definition: str  # "hd" or "sd"
    published_at: str

    @property
    def watch_url(self) -> str:
        return WATCH_URL.format(id=self.video_id)

    @property
    def attribution(self) -> str:
        """The credit string we should put in any short's description."""
        return f'"{self.title}" by {self.channel_title} (licensed CC-BY): {self.watch_url}'


class YouTubeApiError(RuntimeError):
    """Raised when the YouTube Data API rejects a call."""


def parse_iso8601_duration(value: str) -> int:
    """Convert an ISO 8601 duration (``PT9M57S``) to seconds. Returns 0 on parse failure."""
    if not value:
        return 0
    m = _DURATION_RE.match(value.strip())
    if not m:
        return 0
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("mins") or 0)
    seconds = int(m.group("secs") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _api_key() -> str:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeApiError(
            "YOUTUBE_API_KEY is not set. Get one at "
            "https://console.cloud.google.com/apis/credentials and export it."
        )
    return key


def _get(url: str, params: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code != 200:
        snippet = resp.text[:300].replace("\n", " ")
        raise YouTubeApiError(f"{url} -> {resp.status_code}: {snippet}")
    payload: dict[str, Any] = resp.json()
    return payload


def _yt_duration_filter(min_seconds: int) -> str:
    """Return the closest ``videoDuration`` filter for a given minimum seconds value.

    Mapping:
        any    : no filter
        short  : < 240s
        medium : 240-1200s
        long   : > 1200s
    """
    if min_seconds <= 0:
        return "any"
    if min_seconds < 240:
        return "any"
    if min_seconds < 1200:
        return "medium"
    return "long"


def _entry_to_item(
    entry: dict[str, Any],
    *,
    fallback_snippets: dict[str, dict[str, Any]] | None = None,
) -> YouTubeItem | None:
    """Convert a videos.list entry to a :class:`YouTubeItem` or None when malformed."""
    vid = entry.get("id")
    if not vid:
        return None
    snip = entry.get("snippet") or (fallback_snippets or {}).get(vid, {}) or {}
    details = entry.get("contentDetails", {}) or {}
    stats = entry.get("statistics", {}) or {}
    status = entry.get("status", {}) or {}
    duration = parse_iso8601_duration(str(details.get("duration", "")))
    try:
        views = int(stats.get("viewCount", 0))
    except (TypeError, ValueError):
        views = 0
    return YouTubeItem(
        video_id=str(vid),
        title=str(snip.get("title", vid)),
        channel_id=str(snip.get("channelId", "")),
        channel_title=str(snip.get("channelTitle", "")),
        description=str(snip.get("description", "")),
        duration_seconds=duration,
        view_count=views,
        license=str(status.get("license", "")),
        definition=str(details.get("definition", "")),
        published_at=str(snip.get("publishedAt", "")),
    )


def _build_search_params(
    query: str,
    *,
    limit: int,
    min_duration_seconds: int,
    order: str,
    region_code: str | None,
    relevance_language: str | None,
    key: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoLicense": "creativeCommon",
        "order": order,
        "videoDuration": _yt_duration_filter(min_duration_seconds),
        "maxResults": min(50, limit),
        "key": key,
    }
    if region_code:
        params["regionCode"] = region_code
    if relevance_language:
        params["relevanceLanguage"] = relevance_language
    return params


def _collect_search_ids(
    payload: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ids: list[str] = []
    snippets_by_id: dict[str, dict[str, Any]] = {}
    for entry in payload.get("items", []):
        vid = entry.get("id", {}).get("videoId")
        if not vid:
            continue
        ids.append(vid)
        snippets_by_id[vid] = entry.get("snippet", {})
    return ids, snippets_by_id


def _filter_videos(
    payload: dict[str, Any],
    *,
    snippets_by_id: dict[str, dict[str, Any]],
    min_duration_seconds: int,
    max_duration_seconds: int | None,
) -> list[YouTubeItem]:
    out: list[YouTubeItem] = []
    for entry in payload.get("items", []):
        item = _entry_to_item(entry, fallback_snippets=snippets_by_id)
        if item is None or item.license != "creativeCommon":
            continue
        if item.duration_seconds < min_duration_seconds:
            continue
        if max_duration_seconds is not None and item.duration_seconds > max_duration_seconds:
            continue
        out.append(item)
    return out


def search_cc(
    query: str,
    *,
    limit: int = 25,
    min_duration_seconds: int = 60,
    max_duration_seconds: int | None = None,
    order: str = "viewCount",
    region_code: str | None = None,
    relevance_language: str | None = "en",
    timeout: float = 30.0,
) -> list[YouTubeItem]:
    """Return up to ``limit`` Creative Commons-licensed YouTube videos matching ``query``.

    The list is filtered by duration locally (the API only exposes coarse
    short/medium/long buckets) and re-verified to be ``license == "creativeCommon"``.
    """
    if limit <= 0:
        return []

    key = _api_key()
    search_params = _build_search_params(
        query,
        limit=limit,
        min_duration_seconds=min_duration_seconds,
        order=order,
        region_code=region_code,
        relevance_language=relevance_language,
        key=key,
    )
    search_payload = _get(SEARCH_URL, search_params, timeout=timeout)
    ids, snippets_by_id = _collect_search_ids(search_payload)
    if not ids:
        return []

    videos_payload = _get(
        VIDEOS_URL,
        {
            "part": "snippet,contentDetails,statistics,status",
            "id": ",".join(ids),
            "key": key,
        },
        timeout=timeout,
    )
    out = _filter_videos(
        videos_payload,
        snippets_by_id=snippets_by_id,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )

    if order == "viewCount":
        out.sort(key=lambda v: v.view_count, reverse=True)
    return out[:limit]


def lookup(video_id: str, *, timeout: float = 30.0) -> YouTubeItem:
    """Fetch a single video by ID and return a :class:`YouTubeItem`.

    Used when the caller already knows the video they want and just needs
    metadata (e.g. title for filename, license confirmation).
    """
    key = _api_key()
    payload = _get(
        VIDEOS_URL,
        {
            "part": "snippet,contentDetails,statistics,status",
            "id": video_id,
            "key": key,
        },
        timeout=timeout,
    )
    items = payload.get("items", [])
    if not items:
        raise YouTubeApiError(f"YouTube video {video_id!r} not found.")
    item = _entry_to_item(items[0])
    if item is None:
        raise YouTubeApiError(f"YouTube video {video_id!r} returned malformed payload.")
    return item


def download_youtube(
    video_id: str,
    *,
    dest_dir: Path | None = None,
    max_height: int = 1080,
    cookiefile: Path | None = None,
    require_cc: bool = True,
) -> Path:
    """Download a YouTube video using yt-dlp and return the local path.

    By default refuses to download videos whose ``status.license`` is not
    ``creativeCommon`` (set ``require_cc=False`` to override - **only** for
    your own uploads or other content you have rights to).
    """
    if require_cc:
        item = lookup(video_id)
        if item.license != "creativeCommon":
            raise YouTubeApiError(
                f"Refusing to download {video_id}: license={item.license!r}, "
                f"not Creative Commons. Pass require_cc=False if you own this video."
            )

    PATHS.ensure()
    out_dir = dest_dir or (PATHS.downloads / f"yt_{video_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best video <= max_height + best audio, merged into mp4.
    fmt = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"b[height<={max_height}][ext=mp4]/"
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]"
    )
    ydl_opts: dict[str, Any] = {
        "format": fmt,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        "noplaylist": True,
        "retries": 3,
    }
    if cookiefile is not None:
        ydl_opts["cookiefile"] = str(cookiefile)

    # Lazy import so the rest of the module stays test-friendly.
    from yt_dlp import YoutubeDL

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(WATCH_URL.format(id=video_id), download=True)
        # yt-dlp post-processes to mp4; the requested_downloads list has the final path.
        candidates = info.get("requested_downloads") or []
        if candidates:
            return Path(candidates[0]["filepath"])
        # Fallback: scan the output directory for the largest mp4.
        mp4s = sorted(out_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_size, reverse=True)
        if not mp4s:
            raise RuntimeError(f"yt-dlp produced no file for {video_id}")
        return mp4s[0]
