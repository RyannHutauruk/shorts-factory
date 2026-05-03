"""Step 5 - render a 9:16 short with burned-in captions via ffmpeg.

We use the 'blurred-bars' format: the source video is scaled to fit the 1080
width, centered vertically on a 1920-tall canvas whose background is a heavily
blurred copy of the same frame. This preserves the full original frame
(important for movies where focal action is not always centered) while filling
the vertical canvas. Captions are word-by-word from Whisper, burned in with
the `subtitles` filter via an ASS file we generate ourselves.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from .config import PATHS, RENDER
from .transcribe import Segment, Word, words_in_window
from .util import format_timestamp, run, slugify


@dataclass(frozen=True)
class RenderJob:
    source: Path
    start: float
    duration: float
    output: Path
    title: str = ""


def _format_ass_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    return f"{hours:01d}:{minutes:02d}:{secs:05.2f}"


def _ass_header(width: int, height: int) -> str:
    """Build an ASS subtitle header sized for vertical 1080x1920 shorts."""
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Inter,{RENDER.fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,4,2,2,80,80,260,1
Style: Title,Inter,{int(RENDER.fontsize * 1.1)},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,4,2,8,80,80,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _group_words_into_phrases(
    words: list[Word],
    *,
    max_chars: int,
    max_phrase_seconds: float = 2.5,
) -> list[tuple[float, float, str]]:
    """Group word timings into short caption phrases.

    Each phrase is at most `max_chars` characters and `max_phrase_seconds` long,
    so captions stay readable on a phone screen.
    """
    phrases: list[tuple[float, float, str]] = []
    if not words:
        return phrases

    cur_start = words[0].start
    cur_end = words[0].end
    cur_text = words[0].text
    for w in words[1:]:
        candidate = f"{cur_text} {w.text}".strip()
        new_dur = w.end - cur_start
        if len(candidate) <= max_chars and new_dur <= max_phrase_seconds:
            cur_text = candidate
            cur_end = w.end
        else:
            phrases.append((cur_start, cur_end, cur_text))
            cur_start = w.start
            cur_end = w.end
            cur_text = w.text
    phrases.append((cur_start, cur_end, cur_text))
    return phrases


def build_ass(
    segments: list[Segment],
    job: RenderJob,
    *,
    width: int = RENDER.width,
    height: int = RENDER.height,
) -> str:
    """Build an ASS subtitle file string for a single short clip.

    Word timings are clamped to the clip window and shifted so t=0 is the
    beginning of the clip.
    """
    out_lines = [_ass_header(width, height)]

    if job.title:
        # Show the title for the first 3.5 seconds.
        title_text = job.title.replace("\n", " ").strip()
        title_lines = wrap(title_text, width=28) or [title_text]
        title_block = r"\N".join(title_lines)
        out_lines.append(
            f"Dialogue: 0,{_format_ass_time(0.0)},{_format_ass_time(min(3.5, job.duration))},"
            f"Title,,0,0,0,,{title_block}"
        )

    words = words_in_window(segments, job.start, job.start + job.duration)
    # Shift to clip-local time.
    shifted = [
        Word(
            start=max(0.0, w.start - job.start),
            end=min(job.duration, w.end - job.start),
            text=w.text,
        )
        for w in words
        if w.end > job.start and w.start < job.start + job.duration
    ]

    phrases = _group_words_into_phrases(shifted, max_chars=RENDER.max_chars_per_line)
    for start, end, text in phrases:
        if end <= start:
            continue
        safe = text.replace("{", "(").replace("}", ")").replace("\\", "/")
        out_lines.append(
            f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Caption,,0,0,0,,{safe}"
        )
    return "\n".join(out_lines) + "\n"


def _filter_complex(width: int, height: int, ass_path: Path) -> str:
    """Build the 9:16 reframe + caption ffmpeg filtergraph.

    Splits input into two streams:
      * background = scaled-up + blurred + center-cropped to (W,H)
      * foreground = scaled to fit width preserving aspect, padded onto (W,H)
    Then overlays foreground on background and burns the ASS file.
    """
    # ASS path needs to be escaped for ffmpeg's subtitles filter.
    ass_escaped = str(ass_path).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    return (
        f"[0:v]split=2[bg][fg];"
        # Background: scale to cover the canvas, then blur + crop to exact size.
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=20[bgblur];"
        # Foreground: scale to fit the width, add transparent padding to canvas size.
        f"[fg]scale={width}:-2:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0[fgpad];"
        f"[bgblur][fgpad]overlay=0:0:format=auto,"
        f"subtitles=filename='{ass_escaped}'[v]"
    )


def render_clip(
    job: RenderJob,
    segments: list[Segment],
    *,
    width: int = RENDER.width,
    height: int = RENDER.height,
    crf: int = RENDER.crf,
    preset: str = RENDER.preset,
) -> Path:
    """Render a single 9:16 short with burned-in captions.

    Returns the output path. If ffmpeg is missing, raises RuntimeError.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required but not on PATH")

    PATHS.ensure()
    job.output.parent.mkdir(parents=True, exist_ok=True)

    ass_dir = PATHS.work / "ass"
    ass_dir.mkdir(parents=True, exist_ok=True)
    ass_path = ass_dir / f"{slugify(job.output.stem)}.ass"
    ass_path.write_text(build_ass(segments, job, width=width, height=height), encoding="utf-8")

    fc = _filter_complex(width, height, ass_path)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-ss",
        format_timestamp(job.start),
        "-t",
        f"{job.duration:.3f}",
        "-i",
        str(job.source),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        RENDER.audio_bitrate,
        "-shortest",
        "-movflags",
        "+faststart",
        str(job.output),
    ]
    run(cmd)
    return job.output
