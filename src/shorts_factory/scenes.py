"""Step 3 - detect scenes using PySceneDetect's ContentDetector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from .config import PATHS, SCENES


@dataclass(frozen=True)
class Scene:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _cache_path(video_path: Path, threshold: float, min_scene_seconds: float) -> Path:
    PATHS.ensure()
    cache_dir = PATHS.scenes
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{video_path.stem}_t{threshold:.1f}_m{min_scene_seconds:.1f}.json"
    return cache_dir / key


def detect_scenes(
    video_path: Path,
    *,
    threshold: float = SCENES.threshold,
    min_scene_seconds: float = SCENES.min_scene_seconds,
    use_cache: bool = True,
) -> list[Scene]:
    """Run PySceneDetect over `video_path` and return a list of Scenes."""
    cache = _cache_path(video_path, threshold, min_scene_seconds)
    if use_cache and cache.exists():
        raw = json.loads(cache.read_text())
        return [Scene(**s) for s in raw]

    video = open_video(str(video_path))
    fps = video.frame_rate
    min_scene_frames = max(1, int(min_scene_seconds * fps))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_frames))
    sm.detect_scenes(video)
    raw = sm.get_scene_list()
    out: list[Scene] = []
    for i, (start, end) in enumerate(raw):
        out.append(Scene(index=i, start=start.get_seconds(), end=end.get_seconds()))
    if use_cache:
        cache.write_text(json.dumps([asdict(s) for s in out]))
    return out


def coalesce_short_scenes(
    scenes: list[Scene],
    *,
    target_seconds: float,
    max_seconds: float,
) -> list[Scene]:
    """Merge runs of short scenes into ~target_seconds windows.

    PySceneDetect emits one scene per shot, but a 'short' is usually 30-60s
    of consecutive shots. We sweep the scene list and combine adjacent shots
    until the window reaches `target_seconds` or would exceed `max_seconds`.
    """
    if not scenes:
        return []
    merged: list[Scene] = []
    current_start = scenes[0].start
    current_end = scenes[0].end
    idx = 0
    for s in scenes[1:]:
        window = current_end - current_start
        next_window = s.end - current_start
        if window < target_seconds and next_window <= max_seconds:
            current_end = s.end
            continue
        merged.append(Scene(index=idx, start=current_start, end=current_end))
        idx += 1
        current_start = s.start
        current_end = s.end
    merged.append(Scene(index=idx, start=current_start, end=current_end))
    return merged
