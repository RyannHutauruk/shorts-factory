"""Unit tests for the heuristic ranker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from shorts_factory.rank import ScoredScene, _duration_fit, diversify, rank_scenes
from shorts_factory.scenes import Scene
from shorts_factory.transcribe import Segment, Word


def test_duration_fit_peaks_at_target() -> None:
    fits = {d: _duration_fit(d) for d in (5, 20, 40, 60, 90)}
    assert fits[40] >= fits[20]
    assert fits[40] >= fits[60]
    assert fits[40] > fits[5]


def test_diversify_picks_non_adjacent_scenes() -> None:
    # Three scenes; the 1st and 2nd are adjacent in time, 3rd is far away.
    scenes = [
        ScoredScene(
            scene=Scene(index=0, start=0, end=30),
            score=0.9,
            dialogue=0.5,
            audio=0.5,
            duration_fit=0.5,
            motion=0.5,
        ),
        ScoredScene(
            scene=Scene(index=1, start=20, end=50),
            score=0.8,
            dialogue=0.5,
            audio=0.5,
            duration_fit=0.5,
            motion=0.5,
        ),
        ScoredScene(
            scene=Scene(index=2, start=600, end=630),
            score=0.7,
            dialogue=0.5,
            audio=0.5,
            duration_fit=0.5,
            motion=0.5,
        ),
    ]
    chosen = diversify(scenes, top_n=2, min_gap_seconds=120.0)
    assert [c.scene.index for c in chosen] == [0, 2]


def test_rank_scenes_skips_head_and_tail(tmp_path: Path) -> None:
    scenes = [
        Scene(index=0, start=0, end=10),  # in head skip
        Scene(index=1, start=100, end=140),  # body
        Scene(index=2, start=950, end=990),  # in tail skip if total=1000
    ]
    seg = Segment(
        start=100,
        end=140,
        text="hello world",
        words=[
            Word(start=110, end=110.4, text="hello"),
            Word(start=110.5, end=110.9, text="world"),
        ],
    )
    fake_video = tmp_path / "fake.mp4"
    fake_video.write_text("not really a video")
    with (
        patch("shorts_factory.rank._audio_energy", return_value=0.5),
        patch("shorts_factory.rank._motion", return_value=0.3),
    ):
        scored = rank_scenes(
            scenes,
            [seg],
            video_path=fake_video,
            total_duration=1000.0,
        )
    # Only the middle scene should survive head/tail trimming.
    assert len(scored) == 1
    assert scored[0].scene.index == 1
    assert 0.0 <= scored[0].score <= 1.0
