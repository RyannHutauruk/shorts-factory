"""Compose a long-form 16:9 documentary MP4 from script, narration, and B-roll.

Pipeline:
1. Build a flat shot list: each paragraph (cold open + chapter
   transition/beats/closing + outro) is allocated 1+ images that share
   its narration duration.
2. Render shots as silent video on a 1920x1080 canvas with blurred bars
   and a slow Ken Burns zoom.
3. Burn in captions (bottom-third) and chapter title cards (centred,
   shown for the first ~3s of each chapter).
4. Mix narration with optional ducked music bed and mux into the final.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .broll import BrollAsset
from .longform_broll import LongformBrollPlan
from .longform_script import LongformScript
from .longform_voice import LongformChapterAudio, LongformNarration, LongformParagraphSegment

WIDTH = 1920
HEIGHT = 1080
FPS = 30
CAPTION_FONT_SIZE = 38
TITLE_FONT_SIZE = 48
CHAPTER_CARD_FONT_SIZE = 64
CHAPTER_CARD_DURATION = 3.0
CAPTION_MAX_CHARS = 60  # 16:9 fits much wider lines than 9:16

# Maximum time any single image can stay on screen before we cycle to the next.
# Long-form narration paragraphs can run 30-60s; without this cap a single
# image would hold for the whole paragraph which feels like a still slideshow.
MAX_SHOT_DURATION = 5.5
# Minimum shot duration; below this the Ken Burns motion is barely visible.
MIN_SHOT_DURATION = 2.5


@dataclass(frozen=True)
class LongformShot:
    """A single image displayed for ``duration`` seconds with a motion preset."""

    image_path: Path
    duration: float
    start: float
    end: float
    # One of: "zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down".
    # Defaults to "zoom_in" for backwards-compatibility with the original tests.
    motion: str = "zoom_in"


@dataclass(frozen=True)
class LongformAssembleJob:
    """All inputs needed to assemble the final long-form MP4."""

    script: LongformScript
    narration: LongformNarration
    broll: LongformBrollPlan
    out_path: Path
    music_path: Path | None = None
    title_overlay: str | None = None
    show_chapter_cards: bool = True
    chapter_timestamps_in_description: bool = True


# ---------- Shot list builder ----------


_MOTION_PRESETS = (
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
)


def _take_unseen(
    pool: list[BrollAsset],
    *,
    seen_in_section: set[str],
    chosen: list[BrollAsset],
    max_per_segment: int,
) -> bool:
    """Append unseen assets from ``pool`` into ``chosen``. Returns True when full."""
    for a in pool:
        key = str(a.local_path)
        if key in seen_in_section:
            continue
        seen_in_section.add(key)
        chosen.append(a)
        if len(chosen) >= max_per_segment:
            return True
    return False


def _take_any(pool: list[BrollAsset], *, chosen: list[BrollAsset], max_per_segment: int) -> bool:
    """Append assets from ``pool`` regardless of seen state. Returns True when full."""
    for a in pool:
        chosen.append(a)
        if len(chosen) >= max_per_segment:
            return True
    return False


def _pick_assets_for_segment(
    assets: list[BrollAsset],
    fallbacks: list[BrollAsset],
    *,
    seen_in_section: set[str],
    max_per_segment: int = 12,
) -> list[BrollAsset]:
    """Collect unseen assets from primary then fallback pool.

    The previous implementation hard-capped each paragraph at 3 images.
    For long paragraphs (30-60s of narration) that produced 10-20s shots
    which felt static; we now grab the whole pool and let
    ``_allocate_shots`` cycle through them at the per-shot duration cap.
    """
    chosen: list[BrollAsset] = []
    for src in (assets, fallbacks):
        if _take_unseen(
            src, seen_in_section=seen_in_section, chosen=chosen, max_per_segment=max_per_segment
        ):
            return chosen
    if chosen:
        return chosen
    # All candidates already used in this section -> permit reuse.
    for src in (assets, fallbacks):
        if _take_any(src, chosen=chosen, max_per_segment=max_per_segment):
            return chosen
        if chosen:
            return chosen
    return chosen


def _allocate_shots(
    segment: LongformParagraphSegment,
    assets: list[BrollAsset],
    fallbacks: list[BrollAsset],
    *,
    seen_in_section: set[str],
    motion_offset: int = 0,
) -> list[LongformShot]:
    """Allocate shots covering ``segment.duration`` with rotating motion presets.

    Strategy:
      - Start from how many shots the segment can hold given ``MAX_SHOT_DURATION``.
      - Use that many distinct images, cycling through the available pool if
        the pool is smaller than the shot count.
      - Each shot gets a deterministic motion preset based on its absolute index.

    Raises if both ``assets`` and ``fallbacks`` are empty.
    """
    chosen = _pick_assets_for_segment(assets, fallbacks, seen_in_section=seen_in_section)
    if not chosen:
        raise RuntimeError(
            f"no B-roll asset available for paragraph at {segment.start:.1f}s "
            f"(text starts: {segment.text[:60]!r}...)"
        )

    # How many shots fit while keeping each shot >= MIN_SHOT_DURATION and
    # <= MAX_SHOT_DURATION.
    n_shots = max(1, int(segment.duration // MAX_SHOT_DURATION))
    if segment.duration % MAX_SHOT_DURATION >= MIN_SHOT_DURATION:
        n_shots += 1
    # Don't ask for fewer shots than the segment minimally needs.
    n_shots = max(1, n_shots)
    # No point making more shots than would fit at MIN_SHOT_DURATION.
    max_fit = max(1, int(segment.duration / MIN_SHOT_DURATION))
    n_shots = min(n_shots, max_fit)

    per = segment.duration / n_shots
    shots: list[LongformShot] = []
    cursor = segment.start
    for i in range(n_shots):
        asset = chosen[i % len(chosen)]
        # Last shot absorbs any rounding so we hit segment.end exactly.
        dur = (segment.end - cursor) if i == n_shots - 1 else per
        # Guard against degenerate inputs (a 0-duration segment).
        dur = max(MIN_SHOT_DURATION, dur) if n_shots > 1 else max(0.5, dur)
        motion = _MOTION_PRESETS[(motion_offset + i) % len(_MOTION_PRESETS)]
        shots.append(
            LongformShot(
                image_path=asset.local_path,
                duration=dur,
                start=cursor,
                end=cursor + dur,
                motion=motion,
            )
        )
        cursor += dur
    return shots


def build_shot_list(
    script: LongformScript,
    narration: LongformNarration,
    broll: LongformBrollPlan,
) -> list[LongformShot]:
    """Build the flat shot list that exactly covers the narration timeline."""
    shots: list[LongformShot] = []

    # Cold open
    cold_open_seen: set[str] = set()
    shots.extend(
        _allocate_shots(
            narration.cold_open,
            broll.cold_open_assets,
            broll.outro_assets + [a for ch in broll.chapters for a in ch.chapter_assets],
            seen_in_section=cold_open_seen,
            motion_offset=len(shots),
        )
    )

    # Chapters
    for chapter_audio, chapter_broll in zip(narration.chapters, broll.chapters, strict=True):
        chapter_seen: set[str] = set()
        # Map chapter.paragraphs back to: transition, beat[i], closing.
        # The narration module emits in this order: transition (if non-empty),
        # beat_00..beat_NN, closing (if non-empty). So we walk paragraphs and
        # match by position.
        paragraph_idx = 0
        if narration_has_transition(chapter_audio):
            shots.extend(
                _allocate_shots(
                    chapter_audio.paragraphs[paragraph_idx],
                    chapter_broll.chapter_assets,
                    [a for assets in chapter_broll.beat_assets for a in assets]
                    + broll.cold_open_assets,
                    seen_in_section=chapter_seen,
                    motion_offset=len(shots),
                )
            )
            paragraph_idx += 1

        for beat_pos in range(len(chapter_broll.beat_assets)):
            if paragraph_idx >= len(chapter_audio.paragraphs):
                break
            shots.extend(
                _allocate_shots(
                    chapter_audio.paragraphs[paragraph_idx],
                    chapter_broll.beat_assets[beat_pos],
                    chapter_broll.chapter_assets + broll.cold_open_assets,
                    seen_in_section=chapter_seen,
                    motion_offset=len(shots),
                )
            )
            paragraph_idx += 1

        if paragraph_idx < len(chapter_audio.paragraphs):
            shots.extend(
                _allocate_shots(
                    chapter_audio.paragraphs[paragraph_idx],
                    chapter_broll.chapter_assets,
                    [a for assets in chapter_broll.beat_assets for a in assets]
                    + broll.cold_open_assets,
                    seen_in_section=chapter_seen,
                    motion_offset=len(shots),
                )
            )
            paragraph_idx += 1

    # Outro
    outro_seen: set[str] = set()
    shots.extend(
        _allocate_shots(
            narration.outro,
            broll.outro_assets,
            broll.cold_open_assets + [a for ch in broll.chapters for a in ch.chapter_assets],
            seen_in_section=outro_seen,
            motion_offset=len(shots),
        )
    )
    return shots


def narration_has_transition(chapter_audio: LongformChapterAudio) -> bool:
    """The narration module writes the transition as paragraphs[0] when present.

    We can detect that by checking the audio_path filename - the synthesizer
    names the transition file ``transition.wav``. This matches our convention
    in :mod:`niche.longform_voice`.
    """
    if not chapter_audio.paragraphs:
        return False
    return chapter_audio.paragraphs[0].audio_path.name == "transition.wav"


# ---------- Captions + chapter cards ----------


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:01d}:{m:02d}:{s:05.2f}"


def _ass_header() -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Inter,{CAPTION_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,3,1,2,140,140,80,1
Style: Title,Inter,{TITLE_FONT_SIZE},&H0000F0FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,3,1,8,80,80,90,1
Style: ChapterCard,Inter,{CHAPTER_CARD_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,1,0,0,0,100,100,0,0,3,4,2,5,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _split_paragraph_into_phrases(
    text: str, *, start: float, end: float, max_chars: int = CAPTION_MAX_CHARS
) -> list[tuple[float, float, str]]:
    """Split a paragraph into roughly-equal caption lines."""
    words = text.split()
    if not words:
        return []
    phrases: list[list[str]] = []
    current: list[str] = []
    for w in words:
        candidate = " ".join(current + [w])
        if current and len(candidate) > max_chars:
            phrases.append(current)
            current = [w]
        else:
            current.append(w)
    if current:
        phrases.append(current)

    total_chars = sum(len(" ".join(p)) for p in phrases) or 1
    out: list[tuple[float, float, str]] = []
    cursor = start
    duration = end - start
    for p in phrases:
        weight = len(" ".join(p)) / total_chars
        ph_dur = duration * weight
        out.append((cursor, cursor + ph_dur, " ".join(p)))
        cursor += ph_dur
    if out:
        last_start, _, last_text = out[-1]
        out[-1] = (last_start, end, last_text)
    return out


def _ass_safe(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\\", "/")


def build_caption_ass(
    script: LongformScript,
    narration: LongformNarration,
    *,
    title: str | None = None,
    show_chapter_cards: bool = True,
) -> str:
    """Build the full ASS file: title, chapter cards, paragraph captions."""
    body = [_ass_header()]

    # Optional main title (top, first 3s only)
    if title:
        text = _ass_safe(title.replace("\n", " ").strip())
        body.append(f"Dialogue: 0,{_ass_time(0.0)},{_ass_time(3.0)},Title,,0,0,0,,{text}")

    # Chapter title cards (centred, first 3.5s of each chapter)
    if show_chapter_cards:
        for ch_audio in narration.chapters:
            label = _ass_safe(f"Chapter {ch_audio.chapter_index} - {ch_audio.title}")
            start = ch_audio.start
            end = min(ch_audio.end, start + CHAPTER_CARD_DURATION)
            body.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},ChapterCard,,0,0,0,,{label}"
            )

    # Paragraph-level captions (bottom)
    def emit(seg: LongformParagraphSegment) -> None:
        for s, e, t in _split_paragraph_into_phrases(seg.text, start=seg.start, end=seg.end):
            body.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Caption,,0,0,0,,{_ass_safe(t)}")

    emit(narration.cold_open)
    for ch_audio in narration.chapters:
        for p in ch_audio.paragraphs:
            emit(p)
    emit(narration.outro)

    return "\n".join(body) + "\n"


# ---------- ffmpeg compose ----------


def _zoompan_for_motion(motion: str, n_frames: int) -> tuple[str, str, str]:
    """Return (z_expr, x_expr, y_expr) for a Ken Burns motion preset.

    All motions are designed so the on-screen image moves perceptibly across
    the full shot. Zooms ramp from 1.0 -> 1.18; pans hold zoom at ~1.15 and
    slide x/y across the available range.
    """
    # Avoid divide-by-zero in expressions when n_frames == 1.
    safe = max(1, n_frames - 1)
    if motion == "zoom_in":
        z = "min(1+0.0028*on,1.20)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "zoom_out":
        z = "max(1.20-0.0028*on,1.02)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_left":
        z = "1.15"
        x = f"(iw-iw/zoom)*(1 - on/{safe})"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_right":
        z = "1.15"
        x = f"(iw-iw/zoom)*(on/{safe})"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_up":
        z = "1.15"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(1 - on/{safe})"
    elif motion == "pan_down":
        z = "1.15"
        x = "iw/2-(iw/zoom/2)"
        y = f"(ih-ih/zoom)*(on/{safe})"
    else:
        # Fallback: same as zoom_in.
        z = "min(1+0.0028*on,1.20)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    return z, x, y


def _shot_filter(input_idx: int, duration: float, label: str, motion: str = "zoom_in") -> str:
    """Per-image filter: blurred-bars + Ken Burns motion for a 16:9 canvas."""
    n_frames = max(1, int(round(duration * FPS)))
    z, x, y = _zoompan_for_motion(motion, n_frames)
    # The foreground is fit-inside the 1920x1080 canvas (decrease) and then
    # padded; this avoids "padded dimensions smaller than input" errors when
    # an ultra-wide source image is taller than the canvas after a
    # height-only scale.
    return (
        f"[{input_idx}:v]split=2[bg{input_idx}][fg{input_idx}];"
        f"[bg{input_idx}]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},gblur=sigma=22[bgblur{input_idx}];"
        f"[fg{input_idx}]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black@0[fgpad{input_idx}];"
        f"[bgblur{input_idx}][fgpad{input_idx}]overlay=0:0:format=auto[stacked{input_idx}];"
        f"[stacked{input_idx}]zoompan=z='{z}':x='{x}':y='{y}':"
        f"d={n_frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"setsar=1,format=yuv420p,"
        f"trim=duration={duration},setpts=PTS-STARTPTS[{label}]"
    )


def _build_visual_filter(shots: list[LongformShot], ass_path: Path) -> str:
    parts = [
        _shot_filter(i, shot.duration, f"shot{i}", shot.motion) for i, shot in enumerate(shots)
    ]
    concat_inputs = "".join(f"[shot{i}]" for i in range(len(shots)))
    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return (
        ";".join(parts)
        + f";{concat_inputs}concat=n={len(shots)}:v=1:a=0[seq];"
        + f"[seq]subtitles=filename='{ass_escaped}'[v]"
    )


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): "
            + proc.stderr.decode("utf-8", errors="replace")[-2000:]
        )


def assemble_longform(job: LongformAssembleJob, *, work_dir: Path | None = None) -> Path:
    """Render the final long-form MP4. Returns the output path."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required but not on PATH")

    shots = build_shot_list(job.script, job.narration, job.broll)
    if not shots:
        raise RuntimeError("no shots produced - B-roll fetch returned nothing")

    total_dur = sum(s.duration for s in shots)
    work = work_dir or job.out_path.parent / "_assemble"
    work.mkdir(parents=True, exist_ok=True)
    ass_path = work / "captions.ass"
    ass_path.write_text(
        build_caption_ass(
            job.script,
            job.narration,
            title=job.title_overlay,
            show_chapter_cards=job.show_chapter_cards,
        ),
        encoding="utf-8",
    )

    # Stage A: silent visual track.
    # The filter graph for a 10-min documentary easily exceeds Windows'
    # 32,767-char CreateProcess argument cap, so we write it to a file and
    # pass ``-filter_complex_script`` instead of inlining ``-filter_complex``.
    visual_path = work / "visual.mp4"
    visual_filter_path = work / "visual_filter.txt"
    visual_filter_path.write_text(_build_visual_filter(shots, ass_path), encoding="utf-8")
    cmd_a: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for shot in shots:
        cmd_a += ["-loop", "1", "-t", f"{shot.duration:.3f}", "-i", str(shot.image_path)]
    cmd_a += [
        "-filter_complex_script",
        str(visual_filter_path),
        "-map",
        "[v]",
        "-r",
        str(FPS),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "21",
        "-preset",
        "veryfast",
        "-t",
        f"{total_dur:.3f}",
        "-an",
        str(visual_path),
    ]
    _run(cmd_a)

    # Stage B: mix audio + mux.
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd_b: list[str] = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-i",
        str(visual_path),
        "-i",
        str(job.narration.full_audio),
    ]
    if job.music_path and job.music_path.exists():
        # Music ducked by 12dB under voice via sidechaincompress.
        cmd_b += ["-stream_loop", "-1", "-i", str(job.music_path)]
        audio_filter = (
            f"[1:a]volume=1.0,apad=whole_dur={total_dur:.3f},asplit=2[narr][narr_sc];"
            f"[2:a]volume=0.18,atrim=duration={total_dur:.3f},asetpts=PTS-STARTPTS[bg];"
            "[bg][narr_sc]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=400[ducked];"
            "[narr][ducked]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    else:
        audio_filter = f"[1:a]volume=1.0,apad=whole_dur={total_dur:.3f}[a]"
    cmd_b += [
        "-filter_complex",
        audio_filter,
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{total_dur:.3f}",
        str(job.out_path),
    ]
    _run(cmd_b)
    return job.out_path


# ---------- Description / metadata helpers ----------


def chapter_timestamps_block(narration: LongformNarration) -> str:
    """Produce a YouTube-style ``00:00 Chapter title`` block.

    YouTube auto-renders these as clickable chapters in the player when the
    description starts with timestamps.
    """
    lines: list[str] = ["00:00 Cold open"]
    for ch_audio in narration.chapters:
        sec = int(ch_audio.start)
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        if h:
            ts = f"{h}:{m:02d}:{s:02d}"
        else:
            ts = f"{m:02d}:{s:02d}"
        lines.append(f"{ts} Chapter {ch_audio.chapter_index} - {ch_audio.title}")
    return "\n".join(lines)


@dataclass(frozen=True)
class LongformOutputs:
    """Paths of every artefact a successful longform run produces."""

    mp4: Path
    captions_ass: Path
    metadata_txt: Path
    script_json: Path
    attribution_txt: Path
    extras: list[Path] = field(default_factory=list)
