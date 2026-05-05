"""Compose narration + B-roll + captions into a 9:16 MP4 short.

Pipeline:
1. Build per-sentence "shots" by looping a still image for the duration of
   the narration sentence, applying a Ken Burns zoom for motion, and
   composing onto a 1080x1920 canvas with blurred bars.
2. Concat the shots and burn in sentence-level captions (split into short
   readable phrases) using libass via the ``subtitles`` filter.
3. Mix the narration audio (and optional ducked music bed) onto the video.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .broll import BrollAsset
from .script import Script
from .voice import Narration

WIDTH = 1080
HEIGHT = 1920
FPS = 30
CAPTION_FONT_SIZE = 64
TITLE_FONT_SIZE = 78
CAPTION_MAX_CHARS = 26  # per line on phone screen


@dataclass(frozen=True)
class AssembleJob:
    """Inputs for a single short composition."""

    script: Script
    narration: Narration
    visuals: list[BrollAsset]  # one per narrated sentence (hook + beats + payoff)
    out_path: Path
    music_path: Path | None = None
    title_overlay: str | None = None  # short title shown for first 2s


def _ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:01d}:{m:02d}:{s:05.2f}"


def _ass_header() -> str:
    """Subtitle styles: a punchy phone-friendly caption + a smaller title bar."""
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Inter,{CAPTION_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,5,2,2,80,80,330,1
Style: Title,Inter,{TITLE_FONT_SIZE},&H0000F0FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,5,2,8,80,80,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _split_sentence_into_phrases(
    text: str, *, start: float, end: float, max_chars: int = CAPTION_MAX_CHARS
) -> list[tuple[float, float, str]]:
    """Split a sentence into ~3-4 word caption phrases evenly spaced over its
    duration. Used because we don't have word timings yet."""
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


def build_caption_ass(narration: Narration, *, title: str | None = None) -> str:
    """Build the ASS subtitle string for this narration."""
    body = [_ass_header()]
    if title:
        title_text = title.replace("\n", " ").strip()
        title_text = title_text.replace("{", "(").replace("}", ")").replace("\\", "/")
        body.append(f"Dialogue: 0,{_ass_time(0.0)},{_ass_time(2.2)},Title,,0,0,0,,{title_text}")
    for seg in narration.sentences:
        for s, e, text in _split_sentence_into_phrases(seg.text, start=seg.start, end=seg.end):
            safe = text.replace("{", "(").replace("}", ")").replace("\\", "/")
            body.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Caption,,0,0,0,,{safe}")
    return "\n".join(body) + "\n"


def _shot_filter(input_idx: int, duration: float, label: str) -> str:
    """Per-image filter: blurred-bars composition with subtle Ken Burns zoom.

    The image is held for ``duration`` via ``-loop 1``+``-t``. We render the
    9:16 canvas with a blurred copy as the background and the source image
    centered on top, then apply a slow zoompan over the duration so the
    image feels alive instead of static.
    """
    n_frames = max(1, int(round(duration * FPS)))
    return (
        f"[{input_idx}:v]split=2[bg{input_idx}][fg{input_idx}];"
        f"[bg{input_idx}]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},gblur=sigma=22[bgblur{input_idx}];"
        f"[fg{input_idx}]scale={WIDTH}:-2:flags=lanczos,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black@0[fgpad{input_idx}];"
        f"[bgblur{input_idx}][fgpad{input_idx}]overlay=0:0:format=auto[stacked{input_idx}];"
        f"[stacked{input_idx}]zoompan=z='min(1+0.0009*on,1.10)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={n_frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"setsar=1,format=yuv420p,"
        f"trim=duration={duration},setpts=PTS-STARTPTS[{label}]"
    )


def _compute_shot_durations(narration: Narration, *, tail_pad: float = 0.4) -> list[float]:
    """One duration per visual shot. Each shot covers from its sentence start
    until the next sentence's start (so inter-sentence silences are included).
    The final shot extends ``tail_pad`` seconds past the end of the audio so
    the last word has visual breathing room. Sum equals
    ``narration.duration + tail_pad``.
    """
    segs = narration.sentences
    durations: list[float] = []
    for i, seg in enumerate(segs):
        if i + 1 < len(segs):
            next_start = segs[i + 1].start
        else:
            next_start = narration.duration + tail_pad
        durations.append(max(0.05, next_start - seg.start))
    return durations


def _build_visual_filter(durations: list[float], ass_path: Path) -> str:
    """Build the full filter_complex for the visual (silent) track."""
    shots = [_shot_filter(i, d, f"shot{i}") for i, d in enumerate(durations)]
    concat_inputs = "".join(f"[shot{i}]" for i in range(len(durations)))
    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return (
        ";".join(shots)
        + f";{concat_inputs}concat=n={len(durations)}:v=1:a=0[seq];"
        + f"[seq]subtitles=filename='{ass_escaped}'[v]"
    )


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): "
            + proc.stderr.decode("utf-8", errors="replace")[-2000:]
        )


def assemble(job: AssembleJob, *, work_dir: Path | None = None) -> Path:
    """Compose the final MP4. Returns the output path."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required but not on PATH")

    if len(job.visuals) != len(job.narration.sentences):
        raise ValueError(
            f"need exactly one visual per narrated sentence: "
            f"{len(job.visuals)} visuals vs {len(job.narration.sentences)} sentences"
        )
    # Visual track must span the FULL narration WAV (which includes inter-
    # sentence gaps), otherwise -shortest later trims off the trailing audio.
    durations = _compute_shot_durations(job.narration, tail_pad=0.4)
    total_dur = sum(durations)

    work = work_dir or job.out_path.parent / "_assemble"
    work.mkdir(parents=True, exist_ok=True)
    ass_path = work / "captions.ass"
    ass_path.write_text(build_caption_ass(job.narration, title=job.title_overlay), encoding="utf-8")

    # Stage A: silent visual track.
    visual_path = work / "visual.mp4"
    cmd_a: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    for asset, dur in zip(job.visuals, durations, strict=True):
        cmd_a += ["-loop", "1", "-t", f"{dur:.3f}", "-i", str(asset.local_path)]
    cmd_a += [
        "-filter_complex",
        _build_visual_filter(durations, ass_path),
        "-map",
        "[v]",
        "-r",
        str(FPS),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "veryfast",
        "-t",
        f"{total_dur:.3f}",
        "-an",
        str(visual_path),
    ]
    _run(cmd_a)

    # Stage B: mix audio + mux into final.
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
    # Pad the narration audio with silence to match the (slightly longer)
    # visual track so the trailing tail_pad isn't trimmed by -shortest.
    if job.music_path and job.music_path.exists():
        cmd_b += ["-stream_loop", "-1", "-i", str(job.music_path)]
        audio_filter = (
            f"[1:a]volume=1.0,apad=whole_dur={total_dur:.3f}[narr];"
            f"[2:a]volume=0.07,atrim=duration={total_dur:.3f},asetpts=PTS-STARTPTS[bg];"
            "[narr][bg]amix=inputs=2:duration=first:dropout_transition=0[a]"
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
        "-movflags",
        "+faststart",
        str(job.out_path),
    ]
    _run(cmd_b)
    return job.out_path
