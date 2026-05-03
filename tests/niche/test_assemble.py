"""Tests for niche.assemble: phrase splitter, ASS generation."""

from __future__ import annotations

from pathlib import Path

from shorts_factory.niche.assemble import (
    _compute_shot_durations,
    _split_sentence_into_phrases,
    build_caption_ass,
)
from shorts_factory.niche.voice import Narration, NarrationSegment


def test_split_sentence_into_phrases_respects_max_chars() -> None:
    text = "The Hindenburg was a marvel of German engineering."
    phrases = _split_sentence_into_phrases(text, start=0.0, end=4.0, max_chars=20)
    for _start, _end, t in phrases:
        assert len(t) <= 20 + 5  # +slack for last word
    assert sum(len(t) for _, _, t in phrases) >= len(text) - 5


def test_split_sentence_into_phrases_covers_full_window() -> None:
    text = "Three short words."
    phrases = _split_sentence_into_phrases(text, start=2.0, end=4.0)
    assert phrases[0][0] == 2.0
    assert phrases[-1][1] == 4.0


def test_split_sentence_into_phrases_handles_empty() -> None:
    assert _split_sentence_into_phrases("", start=0.0, end=1.0) == []


def test_build_caption_ass_includes_title_and_lines(tmp_path: Path) -> None:
    seg = NarrationSegment(
        text="The Hindenburg was a marvel.",
        audio_path=tmp_path / "a.wav",
        start=0.0,
        end=2.0,
    )
    n = Narration(full_audio=tmp_path / "full.wav", duration=2.0, sentences=[seg])
    out = build_caption_ass(n, title="Hindenburg disaster")
    assert "Style: Caption" in out
    assert "Style: Title" in out
    assert "Hindenburg disaster" in out
    assert "Hindenburg" in out  # caption text


def test_shot_durations_cover_full_audio_plus_tail(tmp_path: Path) -> None:
    """Regression: visual track must span the FULL narration WAV (audio
    duration + tail_pad), otherwise -shortest crops the last sentence."""
    # 3 sentences, audio durations 2.0/3.0/2.5; 0.18s gaps between → audio
    # ends at 0+2.0 + 0.18+3.0 + 0.18+2.5 = 7.86s.
    starts = [0.0, 2.18, 5.36]
    ends = [2.0, 5.18, 7.86]
    segs = [
        NarrationSegment(text=f"s{i}", audio_path=tmp_path / f"a{i}.wav", start=s, end=e)
        for i, (s, e) in enumerate(zip(starts, ends, strict=True))
    ]
    audio_total = 7.86
    n = Narration(full_audio=tmp_path / "full.wav", duration=audio_total, sentences=segs)
    tail_pad = 0.4
    durations = _compute_shot_durations(n, tail_pad=tail_pad)
    assert len(durations) == 3
    # Sum spans the full audio (including inter-sentence gaps) plus the tail.
    assert abs(sum(durations) - (audio_total + tail_pad)) < 1e-6
    # First shot covers up to the next sentence start (includes the trailing gap).
    assert abs(durations[0] - 2.18) < 1e-6
    # Final shot covers from its sentence start to audio end + tail_pad.
    assert abs(durations[-1] - (audio_total + tail_pad - segs[-1].start)) < 1e-6


def test_build_caption_ass_escapes_ass_control_chars(tmp_path: Path) -> None:
    seg = NarrationSegment(
        text="A {weird} line with \\backslashes.",
        audio_path=tmp_path / "a.wav",
        start=0.0,
        end=1.0,
    )
    n = Narration(full_audio=tmp_path / "full.wav", duration=1.0, sentences=[seg])
    out = build_caption_ass(n)
    assert "{weird}" not in out
    assert "\\backslashes" not in out
