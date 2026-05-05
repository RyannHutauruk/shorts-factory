"""YouTube upload module (OAuth2). See ``youtube.py`` for the uploader entry."""

from .youtube import (
    UploadOptions,
    UploadResult,
    YouTubeAuthError,
    YouTubeUploader,
)

__all__ = [
    "YouTubeAuthError",
    "YouTubeUploader",
    "UploadOptions",
    "UploadResult",
]
