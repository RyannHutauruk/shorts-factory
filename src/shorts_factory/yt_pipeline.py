"""End-to-end pipeline variant that sources from YouTube CC-BY videos.

Mirrors :mod:`shorts_factory.pipeline` but starts from YouTube Data API search
+ yt-dlp download instead of archive.org.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import PATHS, SCENES
from .rank import ScoredScene, diversify, rank_scenes
from .render import RenderJob, render_clip
from .scenes import coalesce_short_scenes, detect_scenes
from .transcribe import transcribe
from .util import probe_duration, slugify
from .youtube import YouTubeItem, download_youtube, lookup, search_cc

console = Console()


@dataclass(frozen=True)
class YouTubePipelineResult:
    item: YouTubeItem
    video_path: Path
    shorts: list[Path]
    scored_scenes: list[ScoredScene]
    attribution: str  # ready-to-paste credit for shorts descriptions


def _print_search_table(items: list[YouTubeItem]) -> None:
    table = Table(title="YouTube CC-BY candidates", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("Video ID")
    table.add_column("Title", overflow="fold")
    table.add_column("Channel", overflow="fold")
    table.add_column("Dur", justify="right")
    table.add_column("Views", justify="right")
    table.add_column("HD", justify="right")
    for i, item in enumerate(items):
        mins, secs = divmod(item.duration_seconds, 60)
        table.add_row(
            str(i),
            item.video_id,
            (item.title or "")[:60],
            (item.channel_title or "")[:30],
            f"{mins}:{secs:02d}",
            f"{item.view_count:,}",
            "yes" if item.definition == "hd" else "",
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


def run_youtube_pipeline(
    *,
    query: str = "",
    video_id: str | None = None,
    max_clips: int = 5,
    clip_seconds: float = 40.0,
    pick_index: int = 0,
    min_duration_seconds: int = 300,
    max_duration_seconds: int | None = None,
    max_height: int = 1080,
    order: str = "viewCount",
) -> YouTubePipelineResult:
    """End-to-end pipeline starting from a YouTube CC-BY query or specific video ID."""
    PATHS.ensure()

    # 1) Discovery -----------------------------------------------------------
    if video_id:
        item = lookup(video_id)
        if item.license != "creativeCommon":
            raise RuntimeError(
                f"{video_id} is licensed {item.license!r}, not Creative Commons. Aborting."
            )
        console.print(f"[bold green]Using:[/bold green] {item.video_id} - {item.title}")
    else:
        if not query:
            raise ValueError("Provide either query or video_id.")
        candidates = search_cc(
            query,
            limit=15,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
            order=order,
        )
        if not candidates:
            raise RuntimeError(
                f"No CC-BY YouTube videos for query={query!r}. "
                "Try a broader query or relax the duration filter."
            )
        _print_search_table(candidates)
        item = candidates[min(pick_index, len(candidates) - 1)]
        console.print(f"[bold green]Picked:[/bold green] {item.video_id} - {item.title}")

    # 2) Download ------------------------------------------------------------
    console.rule("[bold]download")
    video_path = download_youtube(item.video_id, max_height=max_height, require_cc=True)
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
    out_dir = PATHS.out / f"yt_{slugify(item.video_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i, s in enumerate(selected):
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

    # Always emit attribution alongside the shorts so the user can copy-paste.
    attribution = item.attribution
    (out_dir / "ATTRIBUTION.txt").write_text(
        f"Source: {item.title}\n"
        f"By: {item.channel_title}\n"
        f"URL: {item.watch_url}\n"
        f"License: Creative Commons - Attribution (reuse allowed with credit)\n\n"
        f"Suggested credit line for short descriptions:\n"
        f"  {attribution}\n",
        encoding="utf-8",
    )
    console.print(f"attribution: {out_dir / 'ATTRIBUTION.txt'}")

    return YouTubePipelineResult(
        item=item,
        video_path=video_path,
        shorts=rendered,
        scored_scenes=selected,
        attribution=attribution,
    )
