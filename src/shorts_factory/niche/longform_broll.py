"""Per-chapter Wikimedia B-roll fetching for long-form videos.

Reuses the shorts-pipeline ``fetch_broll`` helper but:
  - searches once per beat ``visual_hint``,
  - de-duplicates globally so the same image never appears in two chapters,
  - falls back to the chapter title / topic if a beat returns nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .broll import BrollAsset, fetch_broll
from .longform_script import LongformChapter, LongformScript


@dataclass(frozen=True)
class LongformChapterBroll:
    """All assets for a single chapter, indexed by beat position."""

    chapter_index: int
    title: str
    # Per-beat assets. Index aligns with ``chapter.beats``. Each entry is
    # the list of images to show during that beat (usually 1-2).
    beat_assets: list[list[BrollAsset]]
    # Generic chapter-level fallbacks (used for the transition / closing,
    # or when a beat search returned nothing).
    chapter_assets: list[BrollAsset] = field(default_factory=list)

    @property
    def all_assets(self) -> list[BrollAsset]:
        result: list[BrollAsset] = list(self.chapter_assets)
        for assets in self.beat_assets:
            result.extend(assets)
        return result


@dataclass(frozen=True)
class LongformBrollPlan:
    """B-roll for a full long-form script: cold-open + per-chapter + outro."""

    cold_open_assets: list[BrollAsset]
    chapters: list[LongformChapterBroll]
    outro_assets: list[BrollAsset]

    @property
    def all_assets(self) -> list[BrollAsset]:
        out: list[BrollAsset] = list(self.cold_open_assets)
        for ch in self.chapters:
            out.extend(ch.all_assets)
        out.extend(self.outro_assets)
        return out


def _fetch_dedup(
    query: str,
    *,
    dest_dir: Path,
    seen: set[str],
    max_results: int,
    min_dimension: int = 720,
) -> list[BrollAsset]:
    """Wrapper around ``fetch_broll`` that filters out already-used images."""
    if not query.strip():
        return []
    assets = fetch_broll(
        query, dest_dir=dest_dir, max_results=max_results * 2, min_dimension=min_dimension
    )
    out: list[BrollAsset] = []
    for a in assets:
        key = str(a.local_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= max_results:
            break
    return out


def fetch_chapter_broll(
    chapter: LongformChapter,
    *,
    topic: str,
    dest_dir: Path,
    seen: set[str],
    per_beat: int = 4,
    chapter_fallbacks: int = 8,
) -> LongformChapterBroll:
    """Fetch B-roll for one chapter: per-beat search + chapter-level fallback."""
    chapter_dir = dest_dir / f"chapter_{chapter.index:02d}"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    beat_assets: list[list[BrollAsset]] = []
    for beat in chapter.beats:
        assets = _fetch_dedup(
            beat.visual_hint,
            dest_dir=chapter_dir,
            seen=seen,
            max_results=per_beat,
        )
        beat_assets.append(assets)

    chapter_assets: list[BrollAsset] = []
    for query in (chapter.title, chapter.summary, topic):
        chapter_assets += _fetch_dedup(
            query,
            dest_dir=chapter_dir,
            seen=seen,
            max_results=chapter_fallbacks - len(chapter_assets),
        )
        if len(chapter_assets) >= chapter_fallbacks:
            break

    return LongformChapterBroll(
        chapter_index=chapter.index,
        title=chapter.title,
        beat_assets=beat_assets,
        chapter_assets=chapter_assets,
    )


def fetch_longform_broll(
    script: LongformScript,
    *,
    dest_dir: Path,
    per_beat: int = 4,
    chapter_fallbacks: int = 8,
    cold_open_count: int = 6,
    outro_count: int = 4,
) -> LongformBrollPlan:
    """Top-level: gather all B-roll for a long-form script.

    Globally de-duplicates so the same image is never reused in another
    chapter. ``per_beat`` controls how many images each beat tries to
    fetch; ``chapter_fallbacks`` is the per-chapter fallback pool.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()

    # Cold-open assets are sourced from the topic itself (broad imagery).
    cold_open_dir = dest_dir / "cold_open"
    cold_open_assets = _fetch_dedup(
        script.topic,
        dest_dir=cold_open_dir,
        seen=seen,
        max_results=cold_open_count,
    )

    chapters = [
        fetch_chapter_broll(
            ch,
            topic=script.topic,
            dest_dir=dest_dir,
            seen=seen,
            per_beat=per_beat,
            chapter_fallbacks=chapter_fallbacks,
        )
        for ch in script.chapters
    ]

    outro_dir = dest_dir / "outro"
    outro_assets = _fetch_dedup(
        script.topic,
        dest_dir=outro_dir,
        seen=seen,
        max_results=outro_count,
    )

    return LongformBrollPlan(
        cold_open_assets=cold_open_assets,
        chapters=chapters,
        outro_assets=outro_assets,
    )
