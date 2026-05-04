"""Tests for niche.longform_voice (Piper synthesis is mocked).

We monkey-patch ``_synthesize_one`` to write a tiny silent WAV instead of
calling Piper, so the dataclass plumbing (cursors, gaps, chapter
boundaries, voice rotation) can be verified without a Piper install.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from shorts_factory.niche import longform_voice as lv
from shorts_factory.niche.longform_script import (
    LongformBeat,
    LongformChapter,
    LongformScript,
)


def _write_silent_wav(path: Path, *, seconds: float = 0.5, rate: int = 22050) -> None:
    n_frames = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames)


@pytest.fixture
def fake_piper(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Path]]:
    """Patch _synthesize_one in longform_voice's namespace to write silent WAVs.

    Returns a dict tracking which voice each call used so tests can verify
    voice rotation across chapters.
    """
    calls: dict[str, list[Path]] = {"voices": []}

    def fake(text: str, out_path: Path, *, voice_model: Path, length_scale: float) -> None:
        calls["voices"].append(voice_model)
        # Use word count to give roughly realistic durations (2.6 wps).
        secs = max(0.2, len(text.split()) / 2.6)
        _write_silent_wav(out_path, seconds=secs)

    monkeypatch.setattr(lv, "_synthesize_one", fake)
    return calls


@pytest.fixture
def voices(tmp_path: Path) -> list[Path]:
    """Two fake voice paths that exist on disk (content irrelevant — we mocked Piper)."""
    a = tmp_path / "voice_a.onnx"
    b = tmp_path / "voice_b.onnx"
    a.write_bytes(b"")
    b.write_bytes(b"")
    return [a, b]


def _make_script(*, n_chapters: int = 3, beats_per_chapter: int = 3) -> LongformScript:
    chapters = [
        LongformChapter(
            index=i,
            title=f"Chapter {i}",
            summary="s",
            transition=f"transition for chapter {i} that says some stuff",
            beats=[
                LongformBeat(text=f"chapter {i} beat {j} narration text " * 3, visual_hint="x")
                for j in range(beats_per_chapter)
            ],
            closing=f"chapter {i} closing line",
        )
        for i in range(1, n_chapters + 1)
    ]
    return LongformScript(
        topic="Test topic",
        niche="history",
        cold_open="Cold open hook " * 10,
        chapters=chapters,
        outro="Outro lands the lesson " * 5,
    )


def test_resolve_voices_rejects_missing(tmp_path: Path) -> None:
    bogus = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError):
        lv._resolve_voices([bogus])


def test_resolve_voices_returns_provided(voices: list[Path]) -> None:
    assert lv._resolve_voices(voices) == voices


def test_synthesize_longform_produces_full_track(
    tmp_path: Path,
    voices: list[Path],
    fake_piper: dict[str, list[Path]],
) -> None:
    script = _make_script(n_chapters=3, beats_per_chapter=2)
    narration = lv.synthesize_longform(
        script,
        dest_dir=tmp_path / "narration",
        voices=voices,
        inter_paragraph_gap=0.1,
        inter_chapter_gap=0.3,
    )

    assert narration.full_audio.exists()
    assert narration.duration > 0
    assert len(narration.chapters) == 3

    # Cold open + chapter audios + outro all exist.
    assert narration.cold_open.audio_path.exists()
    assert narration.outro.audio_path.exists()
    for ch in narration.chapters:
        assert ch.audio_path.exists()


def test_synthesize_longform_rotates_voices_by_chapter(
    tmp_path: Path,
    voices: list[Path],
    fake_piper: dict[str, list[Path]],
) -> None:
    script = _make_script(n_chapters=4, beats_per_chapter=2)
    narration = lv.synthesize_longform(
        script,
        dest_dir=tmp_path / "narration",
        voices=voices,
    )

    # Round-robin: chapter 1 -> voices[0], chapter 2 -> voices[1],
    # chapter 3 -> voices[0], chapter 4 -> voices[1].
    assert narration.chapters[0].voice_path == voices[0]
    assert narration.chapters[1].voice_path == voices[1]
    assert narration.chapters[2].voice_path == voices[0]
    assert narration.chapters[3].voice_path == voices[1]


def test_synthesize_longform_chapter_timings_are_monotonic(
    tmp_path: Path,
    voices: list[Path],
    fake_piper: dict[str, list[Path]],
) -> None:
    script = _make_script(n_chapters=3, beats_per_chapter=3)
    narration = lv.synthesize_longform(
        script,
        dest_dir=tmp_path / "narration",
        voices=voices,
    )

    # Cold open starts at 0
    assert narration.cold_open.start == 0
    # Each chapter starts at-or-after the prior one's end
    cursor = narration.cold_open.end
    for ch in narration.chapters:
        assert ch.start >= cursor
        # Paragraph segments inside a chapter are also monotonic
        prev_end = ch.start
        for seg in ch.paragraphs:
            assert seg.start >= prev_end - 0.01  # allow tiny rounding
            assert seg.end >= seg.start
            prev_end = seg.end
        cursor = ch.end
    # Outro sits after the last chapter
    assert narration.outro.start >= cursor


def test_synthesize_longform_uses_single_voice_when_only_one_provided(
    tmp_path: Path,
    voices: list[Path],
    fake_piper: dict[str, list[Path]],
) -> None:
    script = _make_script(n_chapters=3, beats_per_chapter=2)
    narration = lv.synthesize_longform(
        script,
        dest_dir=tmp_path / "narration",
        voices=[voices[0]],
    )
    for ch in narration.chapters:
        assert ch.voice_path == voices[0]
