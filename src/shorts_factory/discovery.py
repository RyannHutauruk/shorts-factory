"""Step 1 - search the Internet Archive for public-domain feature films.

We use the JSON advanced-search endpoint; no API key required.
Docs: https://archive.org/advancedsearch.php
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

ADVANCED_SEARCH = "https://archive.org/advancedsearch.php"
DETAIL_URL = "https://archive.org/details/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}"

# Collections that overwhelmingly contain public-domain or permissively-licensed
# feature films / shorts on archive.org.
DEFAULT_COLLECTIONS = (
    "feature_films",
    "publicmovies",
    "classic_tv",
    "moviesandfilms",
)


@dataclass(frozen=True)
class ArchiveItem:
    identifier: str
    title: str
    year: int | None
    avg_rating: float | None
    downloads: int | None
    description: str | None
    runtime: str | None

    @property
    def detail_url(self) -> str:
        return DETAIL_URL.format(identifier=self.identifier)

    @property
    def download_url(self) -> str:
        return DOWNLOAD_URL.format(identifier=self.identifier)


def _build_query(query: str, collections: tuple[str, ...]) -> str:
    coll = " OR ".join(f'collection:"{c}"' for c in collections)
    safe_query = query.replace('"', "")
    return f'({coll}) AND mediatype:"movies" AND ({safe_query})'


def search(
    query: str,
    *,
    limit: int = 20,
    collections: tuple[str, ...] = DEFAULT_COLLECTIONS,
    timeout: float = 30.0,
) -> list[ArchiveItem]:
    """Search archive.org and return up to `limit` items, ranked by downloads."""
    params = [
        ("q", _build_query(query, collections)),
        ("fl[]", "identifier"),
        ("fl[]", "title"),
        ("fl[]", "year"),
        ("fl[]", "avg_rating"),
        ("fl[]", "downloads"),
        ("fl[]", "description"),
        ("fl[]", "runtime"),
        ("sort[]", "downloads desc"),
        ("rows", str(limit)),
        ("page", "1"),
        ("output", "json"),
    ]
    resp = requests.get(ADVANCED_SEARCH, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    docs = payload.get("response", {}).get("docs", [])
    items: list[ArchiveItem] = []
    for d in docs:
        year_raw = d.get("year")
        try:
            year = int(year_raw) if year_raw is not None else None
        except (TypeError, ValueError):
            year = None
        rating_raw = d.get("avg_rating")
        try:
            rating = float(rating_raw) if rating_raw is not None else None
        except (TypeError, ValueError):
            rating = None
        downloads_raw = d.get("downloads")
        try:
            downloads = int(downloads_raw) if downloads_raw is not None else None
        except (TypeError, ValueError):
            downloads = None
        desc = d.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(x) for x in desc)
        items.append(
            ArchiveItem(
                identifier=str(d["identifier"]),
                title=str(d.get("title", d["identifier"])),
                year=year,
                avg_rating=rating,
                downloads=downloads,
                description=str(desc) if desc is not None else None,
                runtime=str(d["runtime"]) if d.get("runtime") else None,
            )
        )
    return items
