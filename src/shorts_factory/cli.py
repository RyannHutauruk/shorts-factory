"""Typer CLI for the shorts-factory pipeline."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import PATHS, SCENES
from .discovery import search as search_archive
from .download import download as download_item
from .niche.batch import BatchOptions, parse_topics_file
from .niche.batch import run_batch as run_niche_batch_v2
from .niche.pipeline import run_niche_pipeline
from .niche.topics import discover_topics, render_topics_text
from .pipeline import run_pipeline
from .rank import diversify, rank_scenes
from .render import RenderJob, render_clip
from .scenes import coalesce_short_scenes, detect_scenes
from .transcribe import transcribe as transcribe_video
from .util import probe_duration
from .youtube import download_youtube as download_yt
from .youtube import search_cc as search_youtube_cc
from .yt_pipeline import run_youtube_pipeline

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


@app.command(name="youtube-search")
def youtube_search(
    query: str = typer.Option(..., "--query", "-q"),
    limit: int = typer.Option(15, "--limit", "-n"),
    min_duration: int = typer.Option(300, "--min-duration", help="Minimum length in seconds."),
    max_duration: int = typer.Option(0, "--max-duration", help="Max length, 0 = unlimited."),
) -> None:
    """Search YouTube for Creative Commons-licensed videos matching a query."""
    items = search_youtube_cc(
        query,
        limit=limit,
        min_duration_seconds=min_duration,
        max_duration_seconds=max_duration or None,
    )
    table = Table(title=f"YouTube CC: {query!r}")
    table.add_column("#", justify="right")
    table.add_column("Video ID")
    table.add_column("Title", overflow="fold")
    table.add_column("Channel", overflow="fold")
    table.add_column("Dur", justify="right")
    table.add_column("Views", justify="right")
    table.add_column("HD", justify="right")
    for i, it in enumerate(items):
        mins, secs = divmod(it.duration_seconds, 60)
        table.add_row(
            str(i),
            it.video_id,
            (it.title or "")[:60],
            (it.channel_title or "")[:30],
            f"{mins}:{secs:02d}",
            f"{it.view_count:,}",
            "yes" if it.definition == "hd" else "",
        )
    console.print(table)


@app.command(name="youtube-download")
def youtube_download(
    video_id: str = typer.Option(..., "--id"),
    max_height: int = typer.Option(1080, "--max-height"),
) -> None:
    """Download a single CC-BY YouTube video by ID."""
    path = download_yt(video_id, max_height=max_height, require_cc=True)
    console.print(f"[green]Downloaded[/green] {path}")


@app.command(name="youtube-run")
def youtube_run(
    query: str = typer.Option("", "--query", "-q"),
    video_id: str = typer.Option("", "--id", help="Skip search and use this YouTube video ID."),
    max_clips: int = typer.Option(5, "--max-clips"),
    clip_seconds: float = typer.Option(40.0, "--clip-seconds"),
    pick_index: int = typer.Option(0, "--pick-index"),
    min_duration: int = typer.Option(300, "--min-duration"),
    max_height: int = typer.Option(1080, "--max-height"),
) -> None:
    """End-to-end pipeline starting from a YouTube CC-BY video."""
    if not query and not video_id:
        raise typer.BadParameter("Provide either --query or --id.")
    result = run_youtube_pipeline(
        query=query,
        video_id=video_id or None,
        max_clips=max_clips,
        clip_seconds=clip_seconds,
        pick_index=pick_index,
        min_duration_seconds=min_duration,
        max_height=max_height,
    )
    console.rule("[bold green]done")
    for p in result.shorts:
        console.print(f"  {p}")
    console.print(f"\n[bold]Required attribution:[/bold] {result.attribution}")


@app.command(name="niche")
def niche_run(
    topic: str = typer.Option(
        "", "--topic", "-t", help="Topic for the explainer (e.g. 'Hindenburg disaster')."
    ),
    niche: str = typer.Option(
        "true_crime", "--niche", help="Niche prompt: true_crime|history|science."
    ),
    script_file: Path = typer.Option(
        None, "--script-file", help="JSON script file (skips Gemini)."
    ),
    music: Path = typer.Option(
        None, "--music", help="Optional background music track (any audio format)."
    ),
    voice: Path = typer.Option(
        None, "--voice", help="Path to a Piper .onnx voice model. Defaults to en_US-ryan-high."
    ),
) -> None:
    """Generate a single faceless-niche narrated short."""
    if not topic and not script_file:
        raise typer.BadParameter("Pass either --topic or --script-file.")
    result = run_niche_pipeline(
        topic=topic,
        niche=niche,
        script_file=script_file,
        music_path=music,
        voice_model=voice,
    )
    console.rule("[bold green]done")
    console.print(f"short:    {result.short_path}")
    console.print(f"metadata: {result.metadata_path}")
    console.print(f"\n[bold]title:[/bold] {result.metadata.title}")
    console.print(f"[bold]description:[/bold] {result.metadata.description}")
    console.print(f"[bold]hashtags:[/bold] {' '.join(result.metadata.hashtags)}")


@app.command(name="niche-batch")
def niche_batch(
    topics_file: Path = typer.Option(None, "--topics-file", help="One topic per line."),
    topic: list[str] = typer.Option([], "--topic", "-t", help="Repeatable. Topics to render."),
    niche: str = typer.Option("true_crime", "--niche"),
    out_root: Path = typer.Option(None, "--out-root", help="Where to write each short's dir."),
    history: Path = typer.Option(
        None, "--history", help="JSON file tracking completed topics + recent hooks."
    ),
    max_count: int = typer.Option(0, "--max-count", help="0 = unlimited."),
) -> None:
    """Generate one short per topic in a list, skipping duplicates and same-stem hooks."""
    topics_list: list[str] = list(topic)
    if topics_file and topics_file.exists():
        topics_list += parse_topics_file(topics_file)
    if not topics_list:
        raise typer.BadParameter("Provide --topic (one or more) or --topics-file.")
    opts = BatchOptions(
        niche=niche,
        out_root=out_root,
        history_path=history,
        max_count=max_count or None,
    )
    results = run_niche_batch_v2(topics_list, options=opts)
    console.rule(f"[bold green]done: {len(results)}/{len(topics_list)} produced")
    for r in results:
        console.print(f"  {r.short_path}")


@app.command(name="topics")
def topics_discover(
    niche: str = typer.Option(
        "history",
        "--niche",
        help="true_crime | history | science | mysteries | weird_facts | biographies | tech_history | space",
    ),
    count: int = typer.Option(30, "--count", "-n"),
    out: Path = typer.Option(None, "--out", help="Append to this queue file. Stdout if unset."),
    avoid_history: Path = typer.Option(
        None, "--avoid-history", help="JSON history file; topics already produced are excluded."
    ),
    audience: str = typer.Option(
        "US", "--audience", help="US | UK | global | general - biases topic selection."
    ),
) -> None:
    """Generate fresh topic ideas in a niche via Gemini."""
    avoid: list[str] = []
    if avoid_history and avoid_history.exists():
        from .niche.batch import BatchHistory

        avoid = BatchHistory.load(avoid_history).topics
    items = discover_topics(niche=niche, count=count, avoid=avoid, audience=audience)
    text = render_topics_text(items)
    if out:
        existing = out.read_text(encoding="utf-8") if out.exists() else ""
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            existing + ("\n" if existing and not existing.endswith("\n") else "") + text,
            encoding="utf-8",
        )
        console.print(f"[green]appended {len(items)} topics to {out}[/green]")
    else:
        console.print(text)


schedule_app = typer.Typer(help="Local scheduler: generate + upload N shorts/day.")
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("init")
def schedule_init(
    config: Path = typer.Option(
        Path("~/.config/shorts-factory/schedule.toml").expanduser(),
        "--config",
        "-c",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Write a starter schedule.toml at the given path."""
    from .upload.scheduler import write_default_config

    write_default_config(config, force=force)
    console.print(f"[green]config:[/green] {config}")


@schedule_app.command("tick")
def schedule_tick(
    config: Path = typer.Option(
        Path("~/.config/shorts-factory/schedule.toml").expanduser(),
        "--config",
        "-c",
    ),
) -> None:
    """Run a single tick (generate one short, upload if enabled). Useful for testing."""
    from .upload.scheduler import load_config, tick_once

    cfg = load_config(config)
    result = tick_once(cfg)
    if result.error:
        console.print(f"[red]error:[/red] {result.error}")
    else:
        console.print(
            f"[green]ok:[/green] {result.topic} -> {result.upload_url or result.short_path}"
        )


@schedule_app.command("run")
def schedule_run(
    config: Path = typer.Option(
        Path("~/.config/shorts-factory/schedule.toml").expanduser(),
        "--config",
        "-c",
    ),
) -> None:
    """Run the scheduler indefinitely (blocking, Ctrl-C to stop)."""
    from .upload.scheduler import run_forever

    run_forever(config)


@schedule_app.command("systemd")
def schedule_systemd(
    config: Path = typer.Option(
        Path("~/.config/shorts-factory/schedule.toml").expanduser(),
        "--config",
        "-c",
    ),
) -> None:
    """Print a sample systemd --user unit file for the scheduler."""
    from .upload.scheduler import systemd_unit

    console.print(systemd_unit(config))


@app.command(name="youtube-upload")
def youtube_upload_cmd(
    video: Path = typer.Option(..., "--video", help="Path to the MP4 to upload."),
    title: str = typer.Option("", "--title"),
    description: str = typer.Option("", "--description"),
    tags: str = typer.Option("", "--tags", help="Comma-separated."),
    privacy: str = typer.Option("public", "--privacy", help="public | unlisted | private"),
    metadata_file: Path = typer.Option(
        None, "--metadata", help="Read title+description+hashtags from a niche METADATA.txt."
    ),
    publish_at: str = typer.Option(
        "", "--publish-at", help="ISO-8601 RFC3339 timestamp; uploads as private and schedules."
    ),
) -> None:
    """Upload a single MP4 to YouTube via the local OAuth flow."""
    from .upload.youtube import UploadOptions, YouTubeUploader

    if metadata_file and metadata_file.exists():
        meta_text = metadata_file.read_text(encoding="utf-8")
        title = title or _extract_section(meta_text, "TITLE")
        description = description or _extract_section(meta_text, "DESCRIPTION")
        if not tags:
            hash_section = _extract_section(meta_text, "HASHTAGS")
            tags = ",".join(t.lstrip("#") for t in hash_section.split() if t)
    if not title:
        raise typer.BadParameter("--title is required (or pass --metadata).")
    opts = UploadOptions(
        title=title,
        description=description,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        privacy_status=privacy,
        publish_at=publish_at or None,
        contains_synthetic_media=True,
    )
    uploader = YouTubeUploader()
    info = uploader.channel_info()
    console.print(f"[bold]channel:[/bold] {info['title']} ({info['id']})")
    result = uploader.upload(video, opts)
    console.rule("[bold green]uploaded")
    console.print(result.url)


def _extract_section(text: str, header: str) -> str:
    """Pull a labelled section from a METADATA.txt-style file."""
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == header:
            in_section = True
            continue
        if in_section:
            if stripped.isupper() and stripped and not stripped.startswith("#"):
                break
            out.append(line)
    return "\n".join(out).strip()


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
