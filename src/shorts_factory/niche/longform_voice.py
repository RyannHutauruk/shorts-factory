"""Long-form Piper TTS: per-chapter synthesis with optional voice rotation.

A 10-minute video read in a single monotone voice is the #1 retention
killer for AI-narrated docs. We address it cheaply by alternating two or
three Piper voices BY CHAPTER (not by sentence - that would feel jarring).

This module reuses the WAV-format helpers from ``niche.voice`` so the
shorts pipeline isn't disturbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .longform_script import LongformChapter, LongformScript
from .voice import (
    _concat_wavs,
    _make_silence,
    _synthesize_one,
    _wav_duration,
    default_voice_path,
)


@dataclass(frozen=True)
class LongformParagraphSegment:
    """One paragraph (transition / beat / closing) inside a chapter, with timing."""

    text: str
    audio_path: Path
    start: float  # seconds within the FULL narration track
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class LongformChapterAudio:
    """Per-chapter audio with paragraph timings."""

    chapter_index: int
    title: str
    audio_path: Path
    duration: float
    start: float  # within the full narration track
    end: float
    paragraphs: list[LongformParagraphSegment]
    voice_path: Path


@dataclass(frozen=True)
class LongformNarration:
    """Full long-form narration (cold-open + chapters + outro) concatenated."""

    full_audio: Path
    duration: float
    cold_open: LongformParagraphSegment
    chapters: list[LongformChapterAudio]
    outro: LongformParagraphSegment


def _resolve_voices(voices: list[Path] | None) -> list[Path]:
    """Validate that all voices exist; default to a single voice if None."""
    if not voices:
        default = default_voice_path()
        if not default.exists():
            raise FileNotFoundError(
                f"piper voice model not found at {default}. "
                f"Download from https://huggingface.co/rhasspy/piper-voices."
            )
        return [default]
    for v in voices:
        if not v.exists():
            raise FileNotFoundError(f"piper voice model not found at {v}")
    return list(voices)


def _synthesize_paragraph(
    text: str,
    out_path: Path,
    *,
    voice: Path,
    length_scale: float,
    cursor: float,
) -> LongformParagraphSegment:
    """Render one paragraph to ``out_path`` and return its timed segment."""
    _synthesize_one(text, out_path, voice_model=voice, length_scale=length_scale)
    dur = _wav_duration(out_path)
    return LongformParagraphSegment(
        text=text,
        audio_path=out_path,
        start=cursor,
        end=cursor + dur,
    )


def synthesize_longform(
    script: LongformScript,
    *,
    dest_dir: Path,
    voices: list[Path] | None = None,
    length_scale: float = 1.0,
    inter_paragraph_gap: float = 0.22,
    inter_chapter_gap: float = 0.6,
) -> LongformNarration:
    """Render a full long-form script and return its timed narration.

    Each chapter uses one voice. Voices rotate round-robin across chapters
    (chapter 1 -> voices[0], chapter 2 -> voices[1], ..., wrapping). The
    cold-open uses ``voices[0]`` and the outro uses ``voices[-1]``.

    Note that all voices must share the same WAV sample format (sample
    rate, channels, bit depth) for concatenation. The ``en_US-*`` Piper
    voices we recommend in the README all match.
    """
    voice_paths = _resolve_voices(voices)
    dest_dir.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    cursor = 0.0

    # Cold open
    cold_open_voice = voice_paths[0]
    cold_open_path = dest_dir / "cold_open.wav"
    cold_open_seg = _synthesize_paragraph(
        script.cold_open,
        cold_open_path,
        voice=cold_open_voice,
        length_scale=length_scale,
        cursor=cursor,
    )
    cursor = cold_open_seg.end
    parts.append(cold_open_path)

    if inter_chapter_gap > 0 and script.chapters:
        gap_path = dest_dir / "gap_cold_open.wav"
        _make_silence(gap_path, inter_chapter_gap, ref_wav=cold_open_path)
        parts.append(gap_path)
        cursor += inter_chapter_gap

    # Chapters
    chapter_audios: list[LongformChapterAudio] = []
    for ch_pos, chapter in enumerate(script.chapters):
        voice = voice_paths[ch_pos % len(voice_paths)]
        chapter_dir = dest_dir / f"chapter_{chapter.index:02d}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        chapter_audio = _synthesize_chapter(
            chapter,
            chapter_dir,
            voice=voice,
            length_scale=length_scale,
            inter_paragraph_gap=inter_paragraph_gap,
            cursor=cursor,
        )
        chapter_audios.append(chapter_audio)
        parts.append(chapter_audio.audio_path)
        cursor = chapter_audio.end

        if inter_chapter_gap > 0 and ch_pos < len(script.chapters) - 1:
            gap_path = dest_dir / f"gap_chapter_{chapter.index:02d}.wav"
            _make_silence(gap_path, inter_chapter_gap, ref_wav=chapter_audio.audio_path)
            parts.append(gap_path)
            cursor += inter_chapter_gap

    # Outro
    outro_voice = voice_paths[-1]
    outro_path = dest_dir / "outro.wav"

    if inter_chapter_gap > 0 and script.chapters:
        gap_path = dest_dir / "gap_outro.wav"
        _make_silence(gap_path, inter_chapter_gap, ref_wav=cold_open_path)
        parts.append(gap_path)
        cursor += inter_chapter_gap

    outro_seg = _synthesize_paragraph(
        script.outro,
        outro_path,
        voice=outro_voice,
        length_scale=length_scale,
        cursor=cursor,
    )
    cursor = outro_seg.end
    parts.append(outro_path)

    full_path = dest_dir / "narration.wav"
    _concat_wavs(parts, full_path)
    total = _wav_duration(full_path)
    return LongformNarration(
        full_audio=full_path,
        duration=total,
        cold_open=cold_open_seg,
        chapters=chapter_audios,
        outro=outro_seg,
    )


def _synthesize_chapter(
    chapter: LongformChapter,
    dest_dir: Path,
    *,
    voice: Path,
    length_scale: float,
    inter_paragraph_gap: float,
    cursor: float,
) -> LongformChapterAudio:
    """Render a single chapter (transition + beats + closing) to a single WAV."""
    chapter_start = cursor
    parts: list[Path] = []
    paragraphs: list[LongformParagraphSegment] = []

    # Build the paragraph list: transition, each beat, closing.
    paragraph_texts: list[tuple[str, str]] = []
    if chapter.transition:
        paragraph_texts.append(("transition", chapter.transition))
    for i, beat in enumerate(chapter.beats):
        paragraph_texts.append((f"beat_{i:02d}", beat.text))
    if chapter.closing:
        paragraph_texts.append(("closing", chapter.closing))

    for idx, (label, text) in enumerate(paragraph_texts):
        chunk_path = dest_dir / f"{label}.wav"
        seg = _synthesize_paragraph(
            text,
            chunk_path,
            voice=voice,
            length_scale=length_scale,
            cursor=cursor,
        )
        paragraphs.append(seg)
        parts.append(chunk_path)
        cursor = seg.end
        # Inter-paragraph gap (except after the last paragraph)
        if idx < len(paragraph_texts) - 1 and inter_paragraph_gap > 0:
            gap_path = dest_dir / f"gap_{idx:02d}.wav"
            _make_silence(gap_path, inter_paragraph_gap, ref_wav=chunk_path)
            parts.append(gap_path)
            cursor += inter_paragraph_gap

    chapter_path = dest_dir / "chapter.wav"
    _concat_wavs(parts, chapter_path)
    duration = _wav_duration(chapter_path)
    return LongformChapterAudio(
        chapter_index=chapter.index,
        title=chapter.title,
        audio_path=chapter_path,
        duration=duration,
        start=chapter_start,
        end=chapter_start + duration,
        paragraphs=paragraphs,
        voice_path=voice,
    )
