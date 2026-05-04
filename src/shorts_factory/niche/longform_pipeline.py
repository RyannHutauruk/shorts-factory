"""End-to-end long-form documentary pipeline.

Topic + niche -> Gemini chapter outline -> per-chapter expansion -> Piper
TTS (multi-voice optional) -> Wikimedia B-roll per chapter -> 16:9 ffmpeg
compose with chapter cards + ducked music bed -> Gemini metadata with
chapter timestamps.

Falls back to ``--script-file`` when no LLM key is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import PATHS
from ..util import slugify
from .broll import BrollAsset
from .longform_assemble import LongformAssembleJob, assemble_longform
from .longform_broll import LongformBrollPlan, fetch_longform_broll
from .longform_metadata import (
    LongformMetadata,
    generate_longform_metadata,
    render_longform_metadata_text,
)
from .longform_script import (
    LongformScript,
    generate_longform_script,
    script_from_file,
    script_to_dict,
)
from .longform_voice import LongformNarration, synthesize_longform


@dataclass(frozen=True)
class LongformResult:
    script: LongformScript
    narration: LongformNarration
    broll: LongformBrollPlan
    metadata: LongformMetadata
    video_path: Path
    metadata_path: Path
    script_json_path: Path
    out_dir: Path

    @property
    def all_visuals(self) -> list[BrollAsset]:
        return self.broll.all_assets


def run_longform_pipeline(
    *,
    topic: str = "",
    niche: str = "history",
    duration_min: float = 10.0,
    audience: str = "US",
    voices: list[Path] | None = None,
    script_file: Path | None = None,
    out_dir: Path | None = None,
    music_path: Path | None = None,
    show_chapter_cards: bool = True,
    skip_metadata: bool = False,
) -> LongformResult:
    """Generate one long-form documentary end-to-end."""
    if not topic and not script_file:
        raise ValueError("must pass either topic or script_file")

    script = (
        script_from_file(script_file)
        if script_file
        else generate_longform_script(
            topic, niche=niche, duration_min=duration_min, audience=audience
        )
    )
    if not topic:
        topic = script.topic

    PATHS.ensure()
    slug = slugify(topic)[:40] or "longform"
    base = out_dir or (PATHS.out / f"longform_{slug}")
    base.mkdir(parents=True, exist_ok=True)

    # Persist the script so the run can be re-rendered without LLM calls.
    script_json_path = base / "script.json"
    script_json_path.write_text(json.dumps(script_to_dict(script), indent=2), encoding="utf-8")

    narration_dir = base / "_narration"
    broll_dir = base / "_broll"
    narration_dir.mkdir(parents=True, exist_ok=True)
    broll_dir.mkdir(parents=True, exist_ok=True)

    print(f"[longform] synthesising narration ({len(script.chapters)} chapters)...")
    narration = synthesize_longform(script, dest_dir=narration_dir, voices=voices)
    print(f"[longform] narration: {narration.duration / 60:.1f} min")

    print(f"[longform] gathering Wikimedia B-roll for {len(script.chapters)} chapters...")
    broll = fetch_longform_broll(script, dest_dir=broll_dir)
    print(f"[longform] {len(broll.all_assets)} B-roll images")

    if skip_metadata:
        meta = LongformMetadata(
            title=f"{topic} - The Untold Story"[:100],
            description=script.cold_open,
            hashtags=[],
            full_description=script.cold_open,
        )
    else:
        print("[longform] generating metadata + chapter timestamps...")
        meta = generate_longform_metadata(script, narration, broll.all_assets)

    video_path = base / f"{slug}.mp4"
    print(f"[longform] composing 16:9 video: {video_path}")
    job = LongformAssembleJob(
        script=script,
        narration=narration,
        broll=broll,
        out_path=video_path,
        music_path=music_path,
        title_overlay=meta.title,
        show_chapter_cards=show_chapter_cards,
    )
    assemble_longform(job, work_dir=base / "_assemble")

    metadata_path = base / "METADATA.txt"
    metadata_path.write_text(render_longform_metadata_text(meta), encoding="utf-8")

    return LongformResult(
        script=script,
        narration=narration,
        broll=broll,
        metadata=meta,
        video_path=video_path,
        metadata_path=metadata_path,
        script_json_path=script_json_path,
        out_dir=base,
    )
