"""Step 4b - heuristic highlight ranker.

We score each (potentially merged) scene on:
  * dialogue density - words spoken per second (normalised to a 0..1 ceiling)
  * audio energy - mean absolute amplitude of the scene window (normalised)
  * duration fit - gaussian-ish bump centred on RANK.target_seconds
  * motion - inter-frame absolute difference proxy (normalised)

The four sub-scores are combined with the weights in `RankSettings`.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import RANK
from .scenes import Scene
from .transcribe import Segment, words_in_window


@dataclass(frozen=True)
class ScoredScene:
    scene: Scene
    score: float
    dialogue: float
    audio: float
    duration_fit: float
    motion: float


def _duration_fit(seconds: float) -> float:
    """Bell-curve around the target duration."""
    target = RANK.target_seconds
    sigma = max(1.0, (RANK.max_seconds - RANK.min_seconds) / 2.0)
    return math.exp(-((seconds - target) ** 2) / (2 * sigma * sigma))


def _audio_energy(video_path: Path, start: float, duration: float) -> float:
    """Mean RMS audio level for the window, in dBFS, mapped to 0..1."""
    if duration <= 0:
        return 0.0
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-af",
        "volumedetect",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = proc.stderr
    mean_db: float | None = None
    for line in out.splitlines():
        if "mean_volume:" in line:
            try:
                mean_db = float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (IndexError, ValueError):
                mean_db = None
    if mean_db is None:
        return 0.0
    # mean_db is typically in [-60, -5] for film audio; map to [0,1].
    norm = (mean_db + 60.0) / 55.0
    return max(0.0, min(1.0, norm))


def _motion(video_path: Path, start: float, duration: float) -> float:
    """Cheap motion proxy: average scene-change-detection score from ffmpeg."""
    if duration <= 0:
        return 0.0
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video_path),
        "-vf",
        "select='gte(scene,0)',metadata=print:file=-",
        "-an",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    scores: list[float] = []
    for line in proc.stdout.splitlines():
        if "scene_score=" in line:
            try:
                scores.append(float(line.split("scene_score=")[1].strip()))
            except (IndexError, ValueError):
                continue
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    return max(0.0, min(1.0, avg * 5.0))


def rank_scenes(
    scenes: list[Scene],
    segments: list[Segment],
    *,
    video_path: Path,
    total_duration: float,
) -> list[ScoredScene]:
    head_cut = total_duration * RANK.skip_head_fraction
    tail_cut = total_duration * (1.0 - RANK.skip_tail_fraction)

    scored: list[ScoredScene] = []
    for scene in scenes:
        if scene.end <= head_cut or scene.start >= tail_cut:
            continue
        dur = scene.duration
        if dur <= 0:
            continue

        words = words_in_window(segments, scene.start, scene.end)
        wps = len(words) / dur if dur > 0 else 0.0
        # Cap at 4 words/sec (rapid dialogue) for normalisation.
        dialogue = max(0.0, min(1.0, wps / 4.0))

        audio = _audio_energy(video_path, scene.start, dur)
        motion = _motion(video_path, scene.start, dur)
        d_fit = _duration_fit(dur)

        score = (
            RANK.weight_dialogue * dialogue
            + RANK.weight_audio_energy * audio
            + RANK.weight_duration_fit * d_fit
            + RANK.weight_motion * motion
        )
        scored.append(
            ScoredScene(
                scene=scene,
                score=score,
                dialogue=dialogue,
                audio=audio,
                duration_fit=d_fit,
                motion=motion,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def diversify(
    scored: list[ScoredScene], *, top_n: int, min_gap_seconds: float = 120.0
) -> list[ScoredScene]:
    """Greedy selection that avoids picking adjacent scenes.

    Walks scored scenes in descending score and only keeps a scene if its
    midpoint is at least `min_gap_seconds` from every previously selected
    scene's midpoint.
    """
    selected: list[ScoredScene] = []
    for s in scored:
        if len(selected) >= top_n:
            break
        mid = (s.scene.start + s.scene.end) / 2.0
        if any(
            abs(mid - (sel.scene.start + sel.scene.end) / 2.0) < min_gap_seconds for sel in selected
        ):
            continue
        selected.append(s)
    return selected
