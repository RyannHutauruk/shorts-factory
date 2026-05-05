"""Tests for niche.broll: junk filter, license check, mocked Wikimedia API."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from shorts_factory.niche.broll import (
    BrollAsset,
    _is_acceptable_license,
    _is_image_mime,
    _looks_like_junk,
    fetch_broll,
)


def test_looks_like_junk_rejects_book_covers() -> None:
    assert _looks_like_junk("Hindenburg book cover.jpg")
    assert _looks_like_junk("Library_catalog_VE23_1928.jpg")
    assert _looks_like_junk("Sheet music for the Hindenburg.png")
    assert _looks_like_junk("Postage stamp commemorating Hindenburg.jpg")


def test_looks_like_junk_accepts_real_photos() -> None:
    assert not _looks_like_junk("Hindenburg disaster 1937.jpg")
    assert not _looks_like_junk("Sam Shere Hindenburg photo.jpg")


def test_is_image_mime() -> None:
    assert _is_image_mime("image/jpeg")
    assert _is_image_mime("image/png")
    assert not _is_image_mime("image/svg+xml")
    assert not _is_image_mime("application/pdf")


def test_is_acceptable_license_rejects_nc() -> None:
    bad = {
        "LicenseShortName": {"value": "CC BY-NC 4.0"},
        "License": {"value": "cc-by-nc-4.0"},
    }
    assert not _is_acceptable_license(bad)


def test_is_acceptable_license_allows_pd() -> None:
    pd = {
        "LicenseShortName": {"value": "Public domain"},
        "License": {"value": "pd"},
    }
    assert _is_acceptable_license(pd)


def test_is_acceptable_license_allows_cc_by_sa() -> None:
    ok = {
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "License": {"value": "cc-by-sa-4.0"},
    }
    assert _is_acceptable_license(ok)


def _fake_wiki_response(pages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"query": {"pages": pages}}


def test_fetch_broll_filters_junk_and_low_res(tmp_path: Path) -> None:
    pages = {
        # OK: high-res PD photo.
        "1": {
            "index": 1,
            "title": "File:Hindenburg disaster 1937.jpg",
            "imageinfo": [
                {
                    "thumbwidth": 1920,
                    "thumbheight": 1500,
                    "mime": "image/jpeg",
                    "thumburl": "https://example.com/h.jpg",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:H.jpg",
                    "extmetadata": {
                        "LicenseShortName": {"value": "Public domain"},
                        "License": {"value": "pd"},
                        "Artist": {"value": "Sam Shere"},
                    },
                }
            ],
        },
        # Junk: book cover.
        "2": {
            "index": 2,
            "title": "File:Hindenburg book cover.jpg",
            "imageinfo": [
                {
                    "thumbwidth": 1920,
                    "thumbheight": 2400,
                    "mime": "image/jpeg",
                    "thumburl": "https://example.com/book.jpg",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Book.jpg",
                    "extmetadata": {
                        "LicenseShortName": {"value": "Public domain"},
                    },
                }
            ],
        },
        # Junk: too small.
        "3": {
            "index": 3,
            "title": "File:Tiny_thumbnail.jpg",
            "imageinfo": [
                {
                    "thumbwidth": 100,
                    "thumbheight": 100,
                    "mime": "image/jpeg",
                    "thumburl": "https://example.com/tiny.jpg",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Tiny.jpg",
                    "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}},
                }
            ],
        },
        # Junk: NC license.
        "4": {
            "index": 4,
            "title": "File:Real_photo_but_nc.jpg",
            "imageinfo": [
                {
                    "thumbwidth": 1920,
                    "thumbheight": 1500,
                    "mime": "image/jpeg",
                    "thumburl": "https://example.com/nc.jpg",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:NC.jpg",
                    "extmetadata": {
                        "LicenseShortName": {"value": "CC BY-NC 4.0"},
                        "License": {"value": "cc-by-nc-4.0"},
                    },
                }
            ],
        },
    }

    class FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return _fake_wiki_response(pages)

    class FakeImageResp:
        status_code = 200
        content = b"\xff\xd8\xff\xe0fake jpg bytes"

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, **_: Any) -> Any:
        if "api.php" in url:
            return FakeResp()
        return FakeImageResp()

    with patch("shorts_factory.niche.broll.requests.get", side_effect=fake_get):
        out = fetch_broll("Hindenburg disaster", dest_dir=tmp_path, max_results=4)

    assert len(out) == 1
    assert out[0].title == "Hindenburg disaster 1937.jpg"
    assert "Public domain" in out[0].license_short
    assert out[0].creator == "Sam Shere"
    assert out[0].local_path.exists()


def test_attribution_line() -> None:
    a = BrollAsset(
        query="x",
        title="Hindenburg disaster.jpg",
        local_path=Path("/tmp/x.jpg"),
        width=1920,
        height=1500,
        mime="image/jpeg",
        license_short="Public domain",
        creator="Sam Shere",
        page_url="https://commons.wikimedia.org/wiki/File:Hindenburg.jpg",
    )
    line = a.attribution_line
    assert "Sam Shere" in line
    assert "Public domain" in line
    assert "commons.wikimedia.org" in line
