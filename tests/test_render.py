"""Unit tests for the ASS-building part of render.py (no ffmpeg invocation)."""

from __future__ import annotations

from pathlib import Path

from shorts_factory.render import RenderJob, _group_words_into_phrases, build_ass
from shorts_factory.transcribe import Segment, Word


def test_group_words_respects_char_limit() -> None:
    words = [
        Word(start=0.0, end=0.3, text="alpha"),
        Word(start=0.4, end=0.7, text="bravo"),
        Word(start=0.8, end=1.1, text="charlie"),
        Word(start=1.2, end=1.5, text="delta"),
    ]
    phrases = _group_words_into_phrases(words, max_chars=12)
    # "alpha bravo" (11 chars) -> ok, "charlie delta" (13) too long, splits
    assert len(phrases) >= 2
    for _start, _end, text in phrases:
        assert len(text) <= 12


def test_build_ass_emits_dialogue_lines(tmp_path: Path) -> None:
    seg = Segment(
        start=0.0,
        end=5.0,
        text="hello world",
        words=[
            Word(start=0.5, end=0.9, text="hello"),
            Word(start=1.0, end=1.4, text="world"),
        ],
    )
    job = RenderJob(
        source=tmp_path / "src.mp4",
        start=0.0,
        duration=5.0,
        output=tmp_path / "out.mp4",
        title="A Title",
    )
    ass = build_ass([seg], job)
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    # Title should be present once
    assert "A Title" in ass
    # Caption text should appear
    assert "hello" in ass.lower()


def test_build_ass_clamps_words_outside_window(tmp_path: Path) -> None:
    seg = Segment(
        start=0.0,
        end=100.0,
        text="far away",
        words=[
            Word(start=0.5, end=0.9, text="early"),
            Word(start=50.0, end=50.4, text="middle"),
            Word(start=99.0, end=99.4, text="late"),
        ],
    )
    job = RenderJob(
        source=tmp_path / "src.mp4",
        start=40.0,
        duration=20.0,
        output=tmp_path / "out.mp4",
    )
    ass = build_ass([seg], job)
    assert "middle" in ass.lower()
    assert "early" not in ass.lower()
    assert "late" not in ass.lower()
