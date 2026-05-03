"""Tests for niche.voice: WAV concat, silence creation, format checks."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from shorts_factory.niche.voice import (
    _concat_wavs,
    _make_silence,
    _wav_duration,
    _wav_format,
)


def _write_wav(path: Path, *, seconds: float, rate: int = 22050, channels: int = 1) -> None:
    n_frames = int(round(seconds * rate))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n_frames * channels)


def test_wav_duration(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    _write_wav(p, seconds=1.5)
    assert abs(_wav_duration(p) - 1.5) < 0.01


def test_wav_format(tmp_path: Path) -> None:
    p = tmp_path / "a.wav"
    _write_wav(p, seconds=0.1, rate=22050, channels=1)
    assert _wav_format(p) == (1, 2, 22050)


def test_concat_wavs_concatenates_durations(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "out.wav"
    _write_wav(a, seconds=1.0)
    _write_wav(b, seconds=0.5)
    _concat_wavs([a, b], out)
    assert abs(_wav_duration(out) - 1.5) < 0.01


def test_concat_wavs_rejects_format_mismatch(tmp_path: Path) -> None:
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "out.wav"
    _write_wav(a, seconds=0.1, rate=22050)
    _write_wav(b, seconds=0.1, rate=44100)
    with pytest.raises(RuntimeError, match="format mismatch"):
        _concat_wavs([a, b], out)


def test_make_silence_matches_ref_format(tmp_path: Path) -> None:
    ref = tmp_path / "ref.wav"
    sil = tmp_path / "sil.wav"
    _write_wav(ref, seconds=0.1, rate=22050)
    _make_silence(sil, 0.5, ref_wav=ref)
    assert _wav_format(sil) == _wav_format(ref)
    assert abs(_wav_duration(sil) - 0.5) < 0.01
