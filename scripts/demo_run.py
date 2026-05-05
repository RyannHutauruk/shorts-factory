"""Run rank + render against an already-downloaded movie file.

Used for the demo run where we want to skip the download step and reuse a
local copy of the source video.

Usage:
    python scripts/demo_run.py work/downloads/clean/notld_480p.mp4 \\
        --title "Night of the Living Dead" --max-clips 5 --clip-seconds 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shorts_factory.config import PATHS, SCENES
from shorts_factory.pipeline import _print_scored_scenes
from shorts_factory.rank import diversify, rank_scenes
from shorts_factory.render import RenderJob, render_clip
from shorts_factory.scenes import coalesce_short_scenes, detect_scenes
from shorts_factory.transcribe import transcribe
from shorts_factory.util import probe_duration, slugify


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--title", default="")
    p.add_argument("--max-clips", type=int, default=5)
    p.add_argument("--clip-seconds", type=float, default=40.0)
    p.add_argument("--out-name", default="demo")
    args = p.parse_args()

    PATHS.ensure()
    print(f"video: {args.video}")
    duration = probe_duration(args.video)
    print(f"duration: {duration:.1f}s")

    print("scenes...")
    raw = detect_scenes(args.video)
    print(f"  {len(raw)} raw scenes")
    merged = coalesce_short_scenes(
        raw,
        target_seconds=args.clip_seconds,
        max_seconds=SCENES.max_scene_seconds,
    )
    print(f"  {len(merged)} merged windows")

    print("transcribe (cached if previously run)...")
    segments = transcribe(args.video)
    print(f"  {len(segments)} segments")

    print("rank scenes...")
    scored = rank_scenes(merged, segments, video_path=args.video, total_duration=duration)
    selected = diversify(
        scored,
        top_n=args.max_clips,
        min_gap_seconds=max(60.0, args.clip_seconds * 1.5),
    )
    _print_scored_scenes(selected)

    out_dir = PATHS.out / slugify(args.out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(selected):
        scene_dur = s.scene.duration
        if scene_dur > args.clip_seconds:
            mid = (s.scene.start + s.scene.end) / 2.0
            start = max(0.0, mid - args.clip_seconds / 2.0)
            dur = args.clip_seconds
        else:
            start = s.scene.start
            dur = scene_dur
        out_path = out_dir / f"short_{i:02d}.mp4"
        print(
            f"  rendering {out_path.name} (start={start:.1f}s dur={dur:.1f}s score={s.score:.3f})"
        )
        render_clip(
            RenderJob(
                source=args.video,
                start=start,
                duration=dur,
                output=out_path,
                title=args.title,
            ),
            segments,
        )

    print("done")


if __name__ == "__main__":
    main()
