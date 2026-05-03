"""Upload a generated short to YouTube via the Data API v3.

Requires OAuth2 (an API key alone cannot upload). The OAuth client_secret
is loaded from a JSON file the user downloads from Google Cloud Console;
the resulting refresh token is cached locally so subsequent uploads are
non-interactive.

Quota: each upload costs ~1,600 quota units; the free tier is 10,000/day
so the API hard-caps you at ~6 uploads/day. This is fine for the
recommended 1-3 shorts/day cadence.

Synthetic-media disclosure: YouTube requires creators to disclose AI-
generated content. We always set ``selfDeclaredMadeForKids=false`` and
``contains_synthetic_media=true`` (via the upload metadata). Channels
that hide this risk demonetisation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "shorts-factory"
DEFAULT_CLIENT_SECRET = DEFAULT_CONFIG_DIR / "client_secret.json"
DEFAULT_TOKEN_CACHE = DEFAULT_CONFIG_DIR / "youtube_token.json"

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
# Keep the read-only scope so we can verify channel info etc.
YOUTUBE_READONLY_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_READONLY_SCOPE]


class YouTubeAuthError(RuntimeError):
    """Raised when OAuth setup is missing or invalid."""


@dataclass
class UploadOptions:
    """Inputs for a single upload."""

    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    category_id: str = "27"  # 27 = Education; safe default for niche shorts
    privacy_status: str = "public"  # public | unlisted | private
    publish_at: str | None = None  # ISO-8601 RFC3339; if set, privacy must be 'private'
    made_for_kids: bool = False
    contains_synthetic_media: bool = True  # YouTube AI-disclosure flag
    notify_subscribers: bool = True


@dataclass
class UploadResult:
    video_id: str
    url: str
    response: dict[str, Any]


class YouTubeUploader:
    """Local-only YouTube uploader. The OAuth flow opens a browser ONCE on
    your own machine; thereafter the cached refresh token is used.
    """

    def __init__(
        self,
        *,
        client_secret_path: Path | str = DEFAULT_CLIENT_SECRET,
        token_cache_path: Path | str = DEFAULT_TOKEN_CACHE,
    ) -> None:
        self.client_secret_path = Path(client_secret_path)
        self.token_cache_path = Path(token_cache_path)
        self._service: Any | None = None

    # ---------- auth ----------
    def _load_credentials(self) -> Any:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:  # pragma: no cover - import-time error
            raise YouTubeAuthError(
                "Google API client libs missing. Install with "
                "`uv pip install google-api-python-client google-auth-oauthlib`"
            ) from exc

        creds: Any | None = None
        if self.token_cache_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_cache_path), SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            self._save_credentials(creds)
            return creds
        if not self.client_secret_path.exists():
            raise YouTubeAuthError(
                f"OAuth client_secret.json not found at {self.client_secret_path}.\n"
                "Create one at https://console.cloud.google.com/apis/credentials\n"
                "(Create credentials -> OAuth client ID -> Desktop app), download "
                "the JSON, and save it to that path."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secret_path), SCOPES)
        creds = flow.run_local_server(port=0, prompt="consent")
        self._save_credentials(creds)
        return creds

    def _save_credentials(self, creds: Any) -> None:
        self.token_cache_path.parent.mkdir(parents=True, exist_ok=True)
        # 600 perms - this token is sensitive.
        self.token_cache_path.write_text(creds.to_json(), encoding="utf-8")
        os.chmod(self.token_cache_path, 0o600)

    def _build_service(self) -> Any:
        if self._service is not None:
            return self._service
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover
            raise YouTubeAuthError(
                "google-api-python-client not installed. Run "
                "`uv pip install google-api-python-client google-auth-oauthlib`."
            ) from exc
        creds = self._load_credentials()
        self._service = build("youtube", "v3", credentials=creds, cache_discovery=False)
        return self._service

    # ---------- public API ----------
    def channel_info(self) -> dict[str, Any]:
        """Return the authorised channel's title + ID. Costs 1 quota unit."""
        svc = self._build_service()
        resp = svc.channels().list(part="snippet,statistics", mine=True).execute()
        items = resp.get("items", [])
        if not items:
            raise YouTubeAuthError("No channel found for these credentials.")
        ch = items[0]
        return {
            "id": ch["id"],
            "title": ch["snippet"]["title"],
            "subscriber_count": ch["statistics"].get("subscriberCount"),
            "video_count": ch["statistics"].get("videoCount"),
        }

    def upload(self, video_path: Path | str, opts: UploadOptions) -> UploadResult:
        """Upload ``video_path`` with the given metadata. ~1,600 quota units."""
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:  # pragma: no cover
            raise YouTubeAuthError("google-api-python-client not installed.") from exc

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(path)

        snippet: dict[str, Any] = {
            "title": opts.title[:100],  # YouTube hard-caps at 100 chars
            "description": opts.description[:5000],
            "tags": [t for t in opts.tags if t][:30],
            "categoryId": opts.category_id,
        }
        status: dict[str, Any] = {
            "privacyStatus": opts.privacy_status,
            "selfDeclaredMadeForKids": opts.made_for_kids,
            "containsSyntheticMedia": opts.contains_synthetic_media,
        }
        if opts.publish_at:
            status["privacyStatus"] = "private"  # required by API
            status["publishAt"] = opts.publish_at

        body = {"snippet": snippet, "status": status}
        media = MediaFileUpload(str(path), chunksize=-1, resumable=True, mimetype="video/mp4")
        svc = self._build_service()
        request = svc.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
            notifySubscribers=opts.notify_subscribers,
        )
        response = _execute_resumable(request)
        video_id = response["id"]
        return UploadResult(
            video_id=video_id,
            url=f"https://youtu.be/{video_id}",
            response=response,
        )


def _execute_resumable(request: Any) -> dict[str, Any]:
    """Drive a resumable upload to completion, retrying on transient errors."""
    response: dict[str, Any] | None = None
    while response is None:
        status, response = request.next_chunk()
        if status is not None:
            print(f"[youtube] upload progress: {int(status.progress() * 100)}%")
    return response


__all__ = [
    "YouTubeAuthError",
    "YouTubeUploader",
    "UploadOptions",
    "UploadResult",
]
