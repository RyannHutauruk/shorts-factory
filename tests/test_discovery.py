"""Unit tests for the archive.org discovery module (network-mocked)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from shorts_factory.discovery import _build_query, search


def test_build_query_includes_collections() -> None:
    q = _build_query("frankenstein", ("feature_films", "publicmovies"))
    assert 'collection:"feature_films"' in q
    assert 'collection:"publicmovies"' in q
    assert "frankenstein" in q
    assert 'mediatype:"movies"' in q


def test_build_query_strips_double_quotes() -> None:
    q = _build_query('a "tricky" query', ("feature_films",))
    assert '"tricky"' not in q
    assert "tricky" in q


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_search_parses_payload() -> None:
    payload = {
        "response": {
            "docs": [
                {
                    "identifier": "abc",
                    "title": "A Movie",
                    "year": "1968",
                    "avg_rating": "4.7",
                    "downloads": "12345",
                    "description": ["one", "two"],
                    "runtime": "01:30",
                },
                {
                    "identifier": "def",
                    "title": "Another",
                    # purposely missing year, rating, downloads to test fallbacks
                },
            ]
        }
    }
    with patch("shorts_factory.discovery.requests.get", return_value=_FakeResp(payload)):
        items = search("zombie", limit=5)
    assert len(items) == 2
    a, b = items
    assert a.identifier == "abc"
    assert a.year == 1968
    assert a.avg_rating == 4.7
    assert a.downloads == 12345
    assert "one" in (a.description or "")
    assert b.year is None and b.avg_rating is None and b.downloads is None
    assert a.detail_url == "https://archive.org/details/abc"
    assert a.download_url == "https://archive.org/download/abc"
