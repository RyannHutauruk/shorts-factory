"""Tests for niche.longform_broll (Wikimedia search is mocked)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from shorts_factory.niche.broll import BrollAsset
from shorts_factory.niche.longform_broll import (
    fetch_chapter_broll,
    fetch_longform_broll,
)
from shorts_factory.niche.longform_script import (
    LongformBeat,
    LongformChapter,
    LongformScript,
)


def _asset(name: str, *, query: str = "q") -> BrollAsset:
    return BrollAsset(
        query=query,
        title=name,
        local_path=Path(f"/tmp/{name}.jpg"),
        width=1920,
        height=1080,
        mime="image/jpeg",
        license_short="CC BY-SA 4.0",
        creator="Anon",
        page_url=f"https://commons.wikimedia.org/wiki/File:{name}",
    )


@pytest.fixture
def mock_fetch(tmp_path: Path) -> Iterator[dict[str, list[list[BrollAsset]]]]:
    """Patch ``fetch_broll`` so each call returns a deterministic asset list.

    We inject a unique asset name based on the query so dedup logic can
    be exercised without hitting Wikimedia.
    """
    calls: dict[str, list[list[BrollAsset]]] = {"by_query": []}

    counter = {"i": 0}

    def fake_fetch_broll(
        query: str,
        *,
        dest_dir: Path,
        max_results: int,
        min_dimension: int = 480,
    ) -> list[BrollAsset]:
        # Return up to ``max_results`` assets named after the query.
        results = []
        for _i in range(min(max_results, 3)):
            counter["i"] += 1
            # Each asset gets a unique name so dedup is the test's concern.
            results.append(_asset(f"{query.replace(' ', '_')}_{counter['i']}", query=query))
        calls["by_query"].append(results)
        return results

    with patch("shorts_factory.niche.longform_broll.fetch_broll", side_effect=fake_fetch_broll):
        yield calls


def _make_chapter(idx: int, beat_hints: list[str]) -> LongformChapter:
    return LongformChapter(
        index=idx,
        title=f"Chapter {idx}",
        summary="some summary",
        transition="t",
        beats=[LongformBeat(text=f"beat {i}", visual_hint=h) for i, h in enumerate(beat_hints)],
        closing="c",
    )


def _make_script(*, n_chapters: int = 3) -> LongformScript:
    return LongformScript(
        topic="Halifax explosion",
        niche="history",
        cold_open="cold",
        chapters=[
            _make_chapter(i, [f"hint_{i}_a", f"hint_{i}_b"]) for i in range(1, n_chapters + 1)
        ],
        outro="outro",
    )


def test_fetch_chapter_broll_calls_per_beat(
    tmp_path: Path, mock_fetch: dict[str, list[list[BrollAsset]]]
) -> None:
    seen: set[str] = set()
    chapter = _make_chapter(1, ["a", "b", "c"])
    plan = fetch_chapter_broll(
        chapter,
        topic="topic",
        dest_dir=tmp_path,
        seen=seen,
        per_beat=2,
        chapter_fallbacks=2,
    )
    # 3 beats searched + at least 1 fallback search (chapter title).
    assert len(plan.beat_assets) == 3
    assert plan.chapter_index == 1
    assert plan.title == "Chapter 1"


def test_fetch_chapter_broll_dedupes_within_chapter(
    tmp_path: Path, mock_fetch: dict[str, list[list[BrollAsset]]]
) -> None:
    """No asset path should appear in two beats within the same chapter."""
    seen: set[str] = set()
    chapter = _make_chapter(1, ["a", "b", "c"])
    plan = fetch_chapter_broll(
        chapter, topic="t", dest_dir=tmp_path, seen=seen, per_beat=2, chapter_fallbacks=2
    )
    paths = [str(a.local_path) for a in plan.all_assets]
    assert len(paths) == len(set(paths))


def test_fetch_longform_broll_dedupes_across_chapters(
    tmp_path: Path,
) -> None:
    """The same asset can't be returned in two different chapters."""
    counter = {"i": 0}

    def fake_fetch_broll(
        query: str,
        *,
        dest_dir: Path,
        max_results: int,
        min_dimension: int = 480,
    ) -> list[BrollAsset]:
        # Always return the SAME 5 named assets regardless of query, to
        # force the dedup logic to do its job.
        results = []
        for i in range(min(max_results, 5)):
            counter["i"] += 1
            results.append(_asset(f"shared_{i}", query=query))
        return results

    script = _make_script(n_chapters=3)
    with patch("shorts_factory.niche.longform_broll.fetch_broll", side_effect=fake_fetch_broll):
        plan = fetch_longform_broll(script, dest_dir=tmp_path, per_beat=2, chapter_fallbacks=2)

    all_paths = [str(a.local_path) for a in plan.all_assets]
    assert len(all_paths) == len(set(all_paths))


def test_fetch_longform_broll_returns_section_assets(
    tmp_path: Path, mock_fetch: dict[str, list[list[BrollAsset]]]
) -> None:
    script = _make_script(n_chapters=2)
    plan = fetch_longform_broll(script, dest_dir=tmp_path)

    assert len(plan.cold_open_assets) >= 1
    assert len(plan.chapters) == 2
    # outro might be empty if dedup ate everything, but cold-open should have results.
    assert len(plan.cold_open_assets) > 0
