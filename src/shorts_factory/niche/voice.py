"""Piper TTS narration with sentence-level timing and word-level captions.

Each script beat (hook, body sentences, payoff) is synthesized to its own
WAV so we know exact start/end timestamps without relying on whisper to
re-segment the same text. Word-level timestamps (for burned-in captions)
come from running faster-whisper on the concatenated narration audio.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from .script import Script

DEFAULT_VOICE_NAME = "en_US-ryan-high.onnx"


def default_voice_path() -> Path:
    """Resolve the Piper voice model location for this machine.

    Search order:
      1. ``$SHORTS_FACTORY_VOICE`` env var (full path).
      2. ``./work/voices/en_US-ryan-high.onnx`` relative to the current
         working directory (the convention used in our README).
      3. ``~/shorts-factory/work/voices/en_US-ryan-high.onnx``.

    The first path that exists wins. If none exist we still return the
    cwd-relative path so the caller's "not found" error message points
    at the place the user is most likely to fix.
    """
    env_path = os.environ.get("SHORTS_FACTORY_VOICE")
    if env_path:
        return Path(env_path).expanduser()

    cwd_voice = Path.cwd() / "work" / "voices" / DEFAULT_VOICE_NAME
    home_voice = Path.home() / "shorts-factory" / "work" / "voices" / DEFAULT_VOICE_NAME

    for candidate in (cwd_voice, home_voice):
        if candidate.exists():
            return candidate
    return cwd_voice


@dataclass(frozen=True)
class NarrationSegment:
    """A single sentence's audio chunk with its timing in the full track."""

    text: str
    audio_path: Path
    start: float  # seconds within the full narration
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Narration:
    """Full narration: per-sentence segments + concatenated track."""

    full_audio: Path
    duration: float
    sentences: list[NarrationSegment]


class PiperNotInstalledError(RuntimeError):
    pass


def _piper_bin() -> str:
    bin_path = shutil.which("piper")
    if not bin_path:
        raise PiperNotInstalledError(
            "piper CLI not found. Install with `uv pip install piper-tts`."
        )
    return bin_path


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def _synthesize_one(
    text: str,
    out_path: Path,
    *,
    voice_model: Path,
    length_scale: float = 1.0,
) -> None:
    """Render a single sentence to ``out_path`` using piper. ``length_scale<1``
    speeds the voice up; ``>1`` slows it down."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _piper_bin(),
        "--model",
        str(voice_model),
        "--output_file",
        str(out_path),
        "--length_scale",
        str(length_scale),
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError("piper failed: " + proc.stderr.decode("utf-8", errors="replace")[:1000])


def _wav_format(path: Path) -> tuple[int, int, int]:
    """(nchannels, sampwidth, framerate) - the bits that must match for concat."""
    with wave.open(str(path), "rb") as wf:
        return (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())


def _concat_wavs(parts: list[Path], out_path: Path) -> None:
    """Concatenate WAV files into ``out_path``. Assumes identical sample format."""
    if not parts:
        raise ValueError("no WAV parts to concat")
    ref_fmt = _wav_format(parts[0])
    all_frames: list[bytes] = []
    for p in parts:
        if _wav_format(p) != ref_fmt:
            raise RuntimeError(
                f"WAV format mismatch between {parts[0]} and {p} ({ref_fmt} vs {_wav_format(p)})"
            )
        with wave.open(str(p), "rb") as wf:
            all_frames.append(wf.readframes(wf.getnframes()))
    nchan, sampw, rate = ref_fmt
    with wave.open(str(out_path), "wb") as wf_out:
        wf_out.setnchannels(nchan)
        wf_out.setsampwidth(sampw)
        wf_out.setframerate(rate)
        for buf in all_frames:
            wf_out.writeframes(buf)


def synthesize(
    script: Script,
    *,
    dest_dir: Path,
    voice_model: Path | None = None,
    length_scale: float = 1.0,
    inter_sentence_gap: float = 0.18,
) -> Narration:
    """Render the full script with Piper and return the timed narration.

    A short silence is inserted between sentences to feel natural.
    """
    voice = voice_model or default_voice_path()
    if not voice.exists():
        raise FileNotFoundError(
            f"piper voice model not found at {voice}. "
            f"Download from https://huggingface.co/rhasspy/piper-voices."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    sentences_text: list[str] = [script.hook]
    sentences_text.extend(b.text for b in script.beats)
    sentences_text.append(script.payoff)

    parts: list[Path] = []
    segments: list[NarrationSegment] = []
    cursor = 0.0
    for i, text in enumerate(sentences_text):
        chunk_path = dest_dir / f"chunk_{i:02d}.wav"
        _synthesize_one(text, chunk_path, voice_model=voice, length_scale=length_scale)
        dur = _wav_duration(chunk_path)
        segments.append(
            NarrationSegment(text=text, audio_path=chunk_path, start=cursor, end=cursor + dur)
        )
        cursor += dur
        parts.append(chunk_path)
        # Optional gap as a silent WAV.
        if i < len(sentences_text) - 1 and inter_sentence_gap > 0:
            gap_path = dest_dir / f"gap_{i:02d}.wav"
            _make_silence(gap_path, inter_sentence_gap, ref_wav=chunk_path)
            parts.append(gap_path)
            cursor += inter_sentence_gap

    full_path = dest_dir / "narration.wav"
    _concat_wavs(parts, full_path)
    total_duration = _wav_duration(full_path)
    return Narration(full_audio=full_path, duration=total_duration, sentences=segments)


def _make_silence(out_path: Path, seconds: float, *, ref_wav: Path) -> None:
    """Write a silent WAV matching the format of ``ref_wav``."""
    with wave.open(str(ref_wav), "rb") as ref:
        params = ref.getparams()
    n_frames = int(round(seconds * params.framerate))
    silence_bytes = b"\x00" * (n_frames * params.sampwidth * params.nchannels)
    with wave.open(str(out_path), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(silence_bytes)
