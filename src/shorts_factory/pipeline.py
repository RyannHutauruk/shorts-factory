"""Glue that runs steps 1->5 end-to-end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import PATHS, SCENES
from .discovery import ArchiveItem, search
from .download import download
from .rank import ScoredScene, diversify, rank_scenes
from .render import RenderJob, render_clip
from .scenes import coalesce_short_scenes, detect_scenes
from .transcribe import transcribe
from .util import probe_duration, slugify

console = Console()


@dataclass(frozen=True)
class PipelineResult:
    item: ArchiveItem
    video_path: Path
    shorts: list[Path]
    scored_scenes: list[ScoredScene]


def _print_search_table(items: list[ArchiveItem]) -> None:
    table = Table(title="Archive.org candidates", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Identifier")
    table.add_column("Title", overflow="fold")
    table.add_column("Year", justify="right")
    table.add_column("Rating", justify="right")
    table.add_column("Downloads", justify="right")
    for i, item in enumerate(items):
        table.add_row(
            str(i),
            item.identifier,
            (item.title or "")[:60],
            str(item.year or ""),
            f"{item.avg_rating:.1f}" if item.avg_rating is not None else "",
            f"{item.downloads:,}" if item.downloads is not None else "",
        )
    console.print(table)


def _print_scored_scenes(scored: list[ScoredScene]) -> None:
    table = Table(title="Scene ranking", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Start")
    table.add_column("Dur", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Dialogue", justify="right")
    table.add_column("Audio", justify="right")
    table.add_column("DurFit", justify="right")
    table.add_column("Motion", justify="right")
    for i, s in enumerate(scored):
        table.add_row(
            str(i),
            f"{s.scene.start:0.1f}",
            f"{s.scene.duration:0.1f}",
            f"{s.score:0.3f}",
            f"{s.dialogue:0.2f}",
            f"{s.audio:0.2f}",
            f"{s.duration_fit:0.2f}",
            f"{s.motion:0.2f}",
        )
    console.print(table)


def run_pipeline(
    *,
    query: str,
    max_clips: int = 5,
    clip_seconds: float = 40.0,
    pick_index: int = 0,
    identifier: str | None = None,
) -> PipelineResult:
    """End-to-end: search -> download -> scenes -> transcribe -> rank -> render.

    Returns a PipelineResult listing the rendered short paths. Intermediate
    artifacts live under `work/` so subsequent runs reuse downloaded video,
    transcripts, etc.
    """
    PATHS.ensure()

    # 1) Discovery -----------------------------------------------------------
    if identifier:
        item = ArchiveItem(
            identifier=identifier,
            title=identifier,
            year=None,
            avg_rating=None,
            downloads=None,
            description=None,
            runtime=None,
        )
    else:
        candidates = search(query, limit=10)
        if not candidates:
            raise RuntimeError(f"No archive.org results for {query!r}")
        _print_search_table(candidates)
        item = candidates[min(pick_index, len(candidates) - 1)]
        console.print(f"[bold green]Picked:[/bold green] {item.identifier} - {item.title}")

    # 2) Download ------------------------------------------------------------
    console.rule("[bold]download")
    video_path = download(item.identifier)
    console.print(f"video: {video_path}")
    total_duration = probe_duration(video_path)
    console.print(f"duration: {total_duration:0.1f}s")

    # 3) Scenes --------------------------------------------------------------
    console.rule("[bold]scene detection")
    raw_scenes = detect_scenes(video_path)
    console.print(f"raw scenes: {len(raw_scenes)}")
    merged = coalesce_short_scenes(
        raw_scenes,
        target_seconds=clip_seconds,
        max_seconds=SCENES.max_scene_seconds,
    )
    console.print(f"merged windows: {len(merged)}")

    # 4a) Transcribe ---------------------------------------------------------
    console.rule("[bold]transcribe")
    segments = transcribe(video_path)
    console.print(f"transcript segments: {len(segments)}")

    # 4b) Rank ---------------------------------------------------------------
    console.rule("[bold]rank scenes")
    scored = rank_scenes(merged, segments, video_path=video_path, total_duration=total_duration)
    selected = diversify(scored, top_n=max_clips, min_gap_seconds=max(60.0, clip_seconds * 1.5))
    _print_scored_scenes(selected)

    # 5) Render --------------------------------------------------------------
    console.rule("[bold]render shorts")
    out_dir = PATHS.out / slugify(item.identifier)
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i, s in enumerate(selected):
        # Trim long windows to the requested clip length, anchored on the centre.
        scene_dur = s.scene.duration
        if scene_dur > clip_seconds:
            mid = (s.scene.start + s.scene.end) / 2.0
            start = max(0.0, mid - clip_seconds / 2.0)
            duration = clip_seconds
        else:
            start = s.scene.start
            duration = scene_dur
        out_path = out_dir / f"short_{i:02d}.mp4"
        job = RenderJob(
            source=video_path,
            start=start,
            duration=duration,
            output=out_path,
            title=item.title,
        )
        console.print(
            f"-> {out_path.name} (start={start:0.1f}s dur={duration:0.1f}s score={s.score:0.3f})"
        )
        render_clip(job, segments)
        rendered.append(out_path)

    return PipelineResult(item=item, video_path=video_path, shorts=rendered, scored_scenes=selected)
