"""Tunable knobs and shared paths for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=_project_root)

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def downloads(self) -> Path:
        return self.work / "downloads"

    @property
    def scenes(self) -> Path:
        return self.work / "scenes"

    @property
    def transcripts(self) -> Path:
        return self.work / "transcripts"

    @property
    def out(self) -> Path:
        return self.root / "out"

    @property
    def models(self) -> Path:
        return self.root / "models"

    def ensure(self) -> None:
        for p in (self.work, self.downloads, self.scenes, self.transcripts, self.out, self.models):
            p.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


@dataclass(frozen=True)
class RenderSettings:
    """9:16 reframe + caption settings."""

    width: int = 1080
    height: int = 1920
    crf: int = 20
    preset: str = "medium"
    audio_bitrate: str = "160k"
    fontsize: int = 56
    font_color: str = "white"
    font_outline: str = "&H80000000"  # ASS BGR with alpha (semi-transparent black)
    max_chars_per_line: int = 24


@dataclass(frozen=True)
class SceneSettings:
    """PySceneDetect tuning."""

    threshold: float = 27.0  # ContentDetector default; lower = more cuts
    min_scene_seconds: float = 8.0
    max_scene_seconds: float = 90.0


@dataclass(frozen=True)
class WhisperSettings:
    """faster-whisper config."""

    model_size: str = "small"  # base/small/medium - small is a good speed/quality tradeoff
    language: str | None = None
    compute_type: str = "int8"  # CPU-friendly
    device: str = "cpu"


@dataclass(frozen=True)
class RankSettings:
    """Heuristic highlight-ranking weights."""

    target_seconds: float = 40.0
    min_seconds: float = 20.0
    max_seconds: float = 60.0
    weight_dialogue: float = 0.45
    weight_audio_energy: float = 0.25
    weight_duration_fit: float = 0.20
    weight_motion: float = 0.10
    # Skip the first N% and last N% of the film (intros, credits)
    skip_head_fraction: float = 0.05
    skip_tail_fraction: float = 0.05


RENDER = RenderSettings()
SCENES = SceneSettings()
WHISPER = WhisperSettings()
RANK = RankSettings()
