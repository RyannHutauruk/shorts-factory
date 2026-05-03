"""Typer CLI for the shorts-factory pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import PATHS, SCENES
from .discovery import search as search_archive
from .download import download as download_item
from .pipeline import run_pipeline
from .rank import diversify, rank_scenes
from .render import RenderJob, render_clip
from .scenes import coalesce_short_scenes, detect_scenes
from .transcribe import transcribe as transcribe_video
from .util import probe_duration

app = typer.Typer(add_completion=False, help="Public-domain movie -> vertical shorts pipeline.")
console = Console()


@app.command()
def search(
    query: str = typer.Option(..., "--query", "-q"),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Search archive.org for public-domain movies."""
    items = search_archive(query, limit=limit)
    table = Table(title=f"archive.org: {query!r}")
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
            (item.title or "")[:70],
            str(item.year or ""),
            f"{item.avg_rating:.1f}" if item.avg_rating is not None else "",
            f"{item.downloads:,}" if item.downloads is not None else "",
        )
    console.print(table)


@app.command()
def download(
    identifier: str = typer.Option(..., "--identifier", "-i"),
) -> None:
    """Download a movie from archive.org."""
    path = download_item(identifier)
    console.print(f"[green]Downloaded[/green] {path}")


@app.command()
def scenes(
    video: Path = typer.Option(..., "--video"),
    threshold: float = typer.Option(SCENES.threshold, "--threshold"),
) -> None:
    """Run scene detection over a local video file."""
    detected = detect_scenes(video, threshold=threshold)
    console.print(f"{len(detected)} raw scenes")
    for s in detected[:50]:
        console.print(f"  {s.index:>4d}  {s.start:>8.2f}  ->  {s.end:>8.2f}  ({s.duration:>6.2f}s)")


@app.command()
def rank(
    video: Path = typer.Option(..., "--video"),
    top: int = typer.Option(5, "--top"),
    clip_seconds: float = typer.Option(40.0, "--clip-seconds"),
) -> None:
    """Score scenes by the heuristic ranker and print the top N."""
    raw = detect_scenes(video)
    merged = coalesce_short_scenes(
        raw, target_seconds=clip_seconds, max_seconds=SCENES.max_scene_seconds
    )
    segs = transcribe_video(video)
    duration = probe_duration(video)
    scored = rank_scenes(merged, segs, video_path=video, total_duration=duration)
    selected = diversify(scored, top_n=top, min_gap_seconds=max(60.0, clip_seconds * 1.5))
    for i, s in enumerate(selected):
        console.print(
            f"#{i}  start={s.scene.start:0.1f}s  dur={s.scene.duration:0.1f}s  "
            f"score={s.score:0.3f} (dlg={s.dialogue:0.2f} aud={s.audio:0.2f} "
            f"fit={s.duration_fit:0.2f} mot={s.motion:0.2f})"
        )


@app.command()
def render(
    video: Path = typer.Option(..., "--video"),
    start: float = typer.Option(..., "--start"),
    duration: float = typer.Option(40.0, "--duration"),
    out: Path = typer.Option(Path("out/manual_short.mp4"), "--out"),
    title: str = typer.Option("", "--title"),
) -> None:
    """Render a single short from `video` starting at `start`."""
    segs = transcribe_video(video)
    job = RenderJob(source=video, start=start, duration=duration, output=out, title=title)
    out_path = render_clip(job, segs)
    console.print(f"[green]Rendered[/green] {out_path}")


@app.command()
def run(
    query: str = typer.Option(..., "--query", "-q"),
    max_clips: int = typer.Option(5, "--max-clips"),
    clip_seconds: float = typer.Option(40.0, "--clip-seconds"),
    pick_index: int = typer.Option(0, "--pick-index", help="Which search result to use."),
    identifier: str = typer.Option(
        "", "--identifier", help="Skip search and use this archive.org id."
    ),
) -> None:
    """End-to-end: search -> download -> scenes -> transcribe -> rank -> render."""
    result = run_pipeline(
        query=query,
        max_clips=max_clips,
        clip_seconds=clip_seconds,
        pick_index=pick_index,
        identifier=identifier or None,
    )
    console.rule("[bold green]done")
    for p in result.shorts:
        console.print(f"  {p}")


@app.command(name="paths")
def show_paths() -> None:
    """Print the working/output directories the pipeline uses."""
    PATHS.ensure()
    console.print(f"work:        {PATHS.work}")
    console.print(f"downloads:   {PATHS.downloads}")
    console.print(f"transcripts: {PATHS.transcripts}")
    console.print(f"models:      {PATHS.models}")
    console.print(f"out:         {PATHS.out}")


if __name__ == "__main__":
    app()
