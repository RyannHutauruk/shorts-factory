"""Unit tests for the scene-coalescing logic (no PySceneDetect dependency)."""

from __future__ import annotations

from shorts_factory.scenes import Scene, coalesce_short_scenes


def _scenes(*pairs: tuple[float, float]) -> list[Scene]:
    return [Scene(index=i, start=s, end=e) for i, (s, e) in enumerate(pairs)]


def test_coalesce_merges_short_scenes_into_target_window() -> None:
    raw = _scenes((0, 5), (5, 10), (10, 15), (15, 25))
    merged = coalesce_short_scenes(raw, target_seconds=20.0, max_seconds=40.0)
    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 25.0


def test_coalesce_respects_max_seconds() -> None:
    raw = _scenes((0, 30), (30, 60), (60, 100))
    merged = coalesce_short_scenes(raw, target_seconds=20.0, max_seconds=40.0)
    assert [m.duration for m in merged] == [30.0, 30.0, 40.0]


def test_coalesce_empty_input() -> None:
    assert coalesce_short_scenes([], target_seconds=20.0, max_seconds=40.0) == []
