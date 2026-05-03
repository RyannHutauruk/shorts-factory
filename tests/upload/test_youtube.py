"""Tests for the YouTube uploader: metadata shape, scheduling, AI disclosure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_factory.upload.youtube import (
    UploadOptions,
    YouTubeAuthError,
    YouTubeUploader,
)


def test_upload_raises_when_client_secret_missing(tmp_path: Path) -> None:
    """No OAuth client_secret.json -> friendly error pointing the user to GCP."""
    uploader = YouTubeUploader(
        client_secret_path=tmp_path / "missing.json",
        token_cache_path=tmp_path / "token.json",
    )
    with pytest.raises(YouTubeAuthError, match="OAuth client_secret"):
        uploader._load_credentials()


def test_upload_options_defaults_set_ai_disclosure_flag() -> None:
    """The synthetic-media disclosure must be on by default for monetization safety."""
    opts = UploadOptions(title="t", description="d")
    assert opts.contains_synthetic_media is True
    assert opts.made_for_kids is False
    assert opts.privacy_status == "public"


def test_upload_truncates_title_and_description(tmp_path: Path) -> None:
    """YouTube hard-caps title at 100 chars and description at 5000 chars."""
    fake_request = MagicMock()
    fake_request.next_chunk.return_value = (None, {"id": "abc123"})

    fake_videos = MagicMock()
    fake_videos.insert.return_value = fake_request

    fake_service = MagicMock()
    fake_service.videos.return_value = fake_videos

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00" * 1024)

    uploader = YouTubeUploader(
        client_secret_path=tmp_path / "ignored.json",
        token_cache_path=tmp_path / "ignored_token.json",
    )
    uploader._service = fake_service  # bypass auth

    long_title = "X" * 200
    long_desc = "Y" * 6000
    opts = UploadOptions(title=long_title, description=long_desc)
    with patch("googleapiclient.http.MediaFileUpload") as media:
        media.return_value = MagicMock()
        result = uploader.upload(video, opts)
    assert result.video_id == "abc123"
    assert result.url == "https://youtu.be/abc123"

    body = fake_videos.insert.call_args.kwargs["body"]
    assert len(body["snippet"]["title"]) == 100
    assert len(body["snippet"]["description"]) == 5000
    # Disclosure flag must be propagated
    assert body["status"]["containsSyntheticMedia"] is True


def test_upload_with_publish_at_forces_private(tmp_path: Path) -> None:
    """publish_at requires privacyStatus=private per YouTube API spec."""
    fake_request = MagicMock()
    fake_request.next_chunk.return_value = (None, {"id": "vid"})
    fake_videos = MagicMock()
    fake_videos.insert.return_value = fake_request
    fake_service = MagicMock()
    fake_service.videos.return_value = fake_videos

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    uploader = YouTubeUploader(
        client_secret_path=tmp_path / "ignored.json",
        token_cache_path=tmp_path / "ignored_token.json",
    )
    uploader._service = fake_service

    opts = UploadOptions(
        title="t",
        description="d",
        privacy_status="public",  # caller asked public
        publish_at="2030-01-01T12:00:00Z",  # but scheduling forces private
    )
    with patch("googleapiclient.http.MediaFileUpload"):
        uploader.upload(video, opts)
    body = fake_videos.insert.call_args.kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["publishAt"] == "2030-01-01T12:00:00Z"


def test_upload_caps_tag_count_at_30(tmp_path: Path) -> None:
    fake_request = MagicMock()
    fake_request.next_chunk.return_value = (None, {"id": "vid"})
    fake_videos = MagicMock()
    fake_videos.insert.return_value = fake_request
    fake_service = MagicMock()
    fake_service.videos.return_value = fake_videos

    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    uploader = YouTubeUploader(
        client_secret_path=tmp_path / "ignored.json",
        token_cache_path=tmp_path / "ignored_token.json",
    )
    uploader._service = fake_service
    opts = UploadOptions(title="t", description="d", tags=[f"tag{i}" for i in range(50)])
    with patch("googleapiclient.http.MediaFileUpload"):
        uploader.upload(video, opts)
    body = fake_videos.insert.call_args.kwargs["body"]
    assert len(body["snippet"]["tags"]) == 30
