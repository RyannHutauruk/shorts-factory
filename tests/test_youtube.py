"""Unit tests for the YouTube CC ingestion module (network mocked)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from shorts_factory.youtube import (
    YouTubeApiError,
    _yt_duration_filter,
    download_youtube,
    lookup,
    parse_iso8601_duration,
    search_cc,
)


def test_parse_iso8601_duration_minutes_seconds() -> None:
    assert parse_iso8601_duration("PT9M57S") == 9 * 60 + 57


def test_parse_iso8601_duration_hours() -> None:
    assert parse_iso8601_duration("PT1H30M15S") == 3600 + 30 * 60 + 15


def test_parse_iso8601_duration_seconds_only() -> None:
    assert parse_iso8601_duration("PT45S") == 45


def test_parse_iso8601_duration_empty_or_garbage() -> None:
    assert parse_iso8601_duration("") == 0
    assert parse_iso8601_duration("not-a-duration") == 0


def test_yt_duration_filter_buckets() -> None:
    assert _yt_duration_filter(0) == "any"
    assert _yt_duration_filter(60) == "any"
    assert _yt_duration_filter(300) == "medium"
    assert _yt_duration_filter(1500) == "long"


class _FakeResp:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text or ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _search_payload(ids: list[str]) -> dict[str, Any]:
    return {
        "items": [
            {
                "id": {"videoId": vid},
                "snippet": {
                    "title": f"Video {vid}",
                    "channelId": f"chan_{vid}",
                    "channelTitle": f"Creator {vid}",
                    "description": f"desc {vid}",
                    "publishedAt": "2020-01-01T00:00:00Z",
                },
            }
            for vid in ids
        ]
    }


def _videos_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": [
            {
                "id": it["id"],
                "snippet": {
                    "title": it.get("title", f"Video {it['id']}"),
                    "channelId": it.get("channelId", f"chan_{it['id']}"),
                    "channelTitle": it.get("channelTitle", f"Creator {it['id']}"),
                    "description": "",
                    "publishedAt": "2020-01-01T00:00:00Z",
                },
                "contentDetails": {
                    "duration": it.get("duration", "PT5M"),
                    "definition": it.get("definition", "hd"),
                },
                "statistics": {"viewCount": str(it.get("views", 0))},
                "status": {
                    "license": it.get("license", "creativeCommon"),
                    "embeddable": it.get("embeddable", True),
                },
            }
            for it in items
        ]
    }


def test_search_cc_filters_non_cc_results() -> None:
    """Even though the API filter is ``creativeCommon``, we re-verify license."""
    search = _search_payload(["aaa", "bbb", "ccc"])
    videos = _videos_payload(
        [
            {"id": "aaa", "license": "creativeCommon", "duration": "PT10M", "views": 1000},
            {"id": "bbb", "license": "youtube", "duration": "PT10M", "views": 5000},
            {"id": "ccc", "license": "creativeCommon", "duration": "PT10M", "views": 2000},
        ]
    )

    def fake_get(url: str, **_: Any) -> _FakeResp:
        if "search" in url:
            return _FakeResp(search)
        return _FakeResp(videos)

    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-key"}):
        with patch("shorts_factory.youtube.requests.get", side_effect=fake_get):
            items = search_cc("query", limit=5)
    ids = [v.video_id for v in items]
    assert "bbb" not in ids
    assert ids == ["ccc", "aaa"]  # sorted by viewCount desc


def test_search_cc_respects_min_duration() -> None:
    search = _search_payload(["aaa", "bbb"])
    videos = _videos_payload(
        [
            {"id": "aaa", "duration": "PT30S"},  # 30s (too short)
            {"id": "bbb", "duration": "PT5M"},  # 300s (fine)
        ]
    )

    def fake_get(url: str, **_: Any) -> _FakeResp:
        if "search" in url:
            return _FakeResp(search)
        return _FakeResp(videos)

    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-key"}):
        with patch("shorts_factory.youtube.requests.get", side_effect=fake_get):
            items = search_cc("q", limit=5, min_duration_seconds=60)
    assert [v.video_id for v in items] == ["bbb"]


def test_search_cc_raises_without_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(YouTubeApiError):
            search_cc("anything", limit=1)


def test_search_cc_handles_api_error() -> None:
    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-key"}):
        bad = _FakeResp({"error": {"code": 403}}, status_code=403, text="quotaExceeded")
        with patch("shorts_factory.youtube.requests.get", return_value=bad):
            with pytest.raises(YouTubeApiError, match="403"):
                search_cc("q", limit=1)


def test_lookup_returns_typed_item() -> None:
    payload = _videos_payload(
        [
            {
                "id": "xyz",
                "title": "My Video",
                "channelTitle": "Creator",
                "duration": "PT12M34S",
                "views": 99,
                "license": "creativeCommon",
                "definition": "hd",
            }
        ]
    )
    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-key"}):
        with patch("shorts_factory.youtube.requests.get", return_value=_FakeResp(payload)):
            item = lookup("xyz")
    assert item.video_id == "xyz"
    assert item.duration_seconds == 12 * 60 + 34
    assert item.license == "creativeCommon"
    assert "youtube.com/watch?v=xyz" in item.watch_url
    assert "Creator" in item.attribution
    assert "youtube.com/watch?v=xyz" in item.attribution


def test_download_youtube_refuses_non_cc(tmp_path: Any) -> None:
    """Default ``require_cc=True`` should refuse a non-CC video without calling yt-dlp."""
    payload = _videos_payload([{"id": "abc", "license": "youtube"}])
    with patch.dict(os.environ, {"YOUTUBE_API_KEY": "test-key"}):
        with patch("shorts_factory.youtube.requests.get", return_value=_FakeResp(payload)):
            with pytest.raises(YouTubeApiError, match="Refusing to download"):
                download_youtube("abc", dest_dir=tmp_path)
