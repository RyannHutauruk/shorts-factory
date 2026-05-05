"""End-to-end faceless-niche pipeline orchestrator.

Topic / niche -> Gemini script -> Piper TTS narration -> Wikimedia B-roll ->
ffmpeg compose -> Gemini metadata. Falls back to ``--script-file`` when no
LLM key is available.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import PATHS
from ..util import slugify
from .assemble import AssembleJob, assemble
from .broll import BrollAsset, fetch_broll
from .metadata import ShortMetadata, generate_metadata, render_metadata_text
from .script import Script, generate_script, script_from_file
from .voice import Narration, synthesize


@dataclass(frozen=True)
class NicheResult:
    script: Script
    narration: Narration
    visuals: list[BrollAsset]
    metadata: ShortMetadata
    short_path: Path
    metadata_path: Path
    out_dir: Path


def _broaden_query(query: str) -> list[str]:
    """Generate broader fallback queries from a specific phrase.

    e.g. 'SS Edmund Fitzgerald sinking' -> ['Edmund Fitzgerald sinking',
    'Edmund Fitzgerald', 'sinking']. Improves recall when Wikimedia has
    only generic photos of a subject.
    """
    cleaned = re.sub(r"\b(SS|HMS|USS|RMS|MS)\b", "", query, flags=re.IGNORECASE).strip()
    parts = [w for w in cleaned.split() if w]
    out: list[str] = []
    if cleaned and cleaned != query:
        out.append(cleaned)
    if len(parts) >= 2:
        out.append(" ".join(parts[:2]))
    if len(parts) >= 3:
        out.append(" ".join(parts[:3]))
    return out


def _collect_queries(visual_hint: str, topic: str, text: str) -> list[str]:
    """Build the de-duplicated query list to try, broadest variants included."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (visual_hint, topic, text[:60]):
        if not raw:
            continue
        for q in [raw, *_broaden_query(raw)]:
            cleaned = q.strip()
            if cleaned and cleaned.lower() not in seen:
                out.append(cleaned)
                seen.add(cleaned.lower())
    return out


def _try_queries(
    queries: list[str],
    *,
    used_paths: set[Path],
    dest_dir: Path,
    cache: dict[str, list[BrollAsset]],
) -> BrollAsset | None:
    for q in queries:
        if q not in cache:
            try:
                cache[q] = fetch_broll(q, dest_dir=dest_dir, max_results=4)
            except Exception:
                cache[q] = []
        for asset in cache[q]:
            if asset.local_path not in used_paths:
                used_paths.add(asset.local_path)
                return asset
    return None


def _pick_visual_for_sentence(
    text: str,
    visual_hint: str,
    topic: str,
    *,
    used_paths: set[Path],
    found_so_far: list[BrollAsset],
    dest_dir: Path,
    cache: dict[str, list[BrollAsset]],
) -> BrollAsset | None:
    """Try the explicit visual_hint first, then fall back to broader queries
    and finally to recycling already-found assets so a niche short never
    fails just because one beat was over-specific."""
    queries = _collect_queries(visual_hint, topic, text)
    asset = _try_queries(queries, used_paths=used_paths, dest_dir=dest_dir, cache=cache)
    if asset is not None:
        return asset

    # Recycle any already-used asset rather than fail the whole pipeline.
    if found_so_far:
        return found_so_far[len(used_paths) % len(found_so_far)]

    # Last resort: any asset we ever saw, even if used.
    for q in queries:
        for cached in cache.get(q, []):
            return cached
    return None


def _gather_visuals(
    script: Script,
    *,
    broll_dir: Path,
) -> list[BrollAsset]:
    used: set[Path] = set()
    cache: dict[str, list[BrollAsset]] = {}
    sentences: list[tuple[str, str]] = [(script.hook, script.topic)]
    for beat in script.beats:
        sentences.append((beat.text, beat.visual_hint))
    sentences.append((script.payoff, script.topic))

    visuals: list[BrollAsset] = []
    for text, hint in sentences:
        a = _pick_visual_for_sentence(
            text,
            hint,
            script.topic,
            used_paths=used,
            found_so_far=visuals,
            dest_dir=broll_dir,
            cache=cache,
        )
        if a is None:
            raise RuntimeError(
                f"no Wikimedia B-roll found for any of: hint={hint!r}, topic={script.topic!r}. "
                f"Try a broader visual_hint or topic."
            )
        visuals.append(a)
    return visuals


def run_niche_pipeline(
    *,
    topic: str = "",
    niche: str = "true_crime",
    script_file: Path | None = None,
    out_dir: Path | None = None,
    music_path: Path | None = None,
    voice_model: Path | None = None,
    skip_metadata: bool = False,
) -> NicheResult:
    """Generate a single faceless-niche short end-to-end."""
    if not topic and not script_file:
        raise ValueError("must pass either topic or script_file")

    script = script_from_file(script_file) if script_file else generate_script(topic, niche=niche)
    if not topic:
        topic = script.topic or "manual"

    PATHS.ensure()
    slug = slugify(topic)[:40] or "short"
    base = out_dir or (PATHS.out / f"niche_{slug}")
    base.mkdir(parents=True, exist_ok=True)

    # Persist the script so the run can be reproduced (or just re-rendered)
    # without another Gemini call.
    (base / "script.json").write_text(json.dumps(asdict(script), indent=2), encoding="utf-8")

    narration_dir = base / "_narration"
    broll_dir = base / "_broll"
    narration_dir.mkdir(parents=True, exist_ok=True)
    broll_dir.mkdir(parents=True, exist_ok=True)

    narration = synthesize(script, dest_dir=narration_dir, voice_model=voice_model)
    visuals = _gather_visuals(script, broll_dir=broll_dir)

    if skip_metadata:
        # Stub metadata so the rest of the flow still runs (used in tests).
        meta = ShortMetadata(
            title=topic[:60],
            description=script.hook + " " + script.payoff,
            hashtags=[],
            full_description=script.hook + " " + script.payoff,
        )
    else:
        meta = generate_metadata(script, visuals)

    short_path = base / f"{slug}.mp4"
    job = AssembleJob(
        script=script,
        narration=narration,
        visuals=visuals,
        out_path=short_path,
        music_path=music_path,
        title_overlay=meta.title,
    )
    assemble(job, work_dir=base / "_assemble")

    metadata_path = base / "METADATA.txt"
    metadata_path.write_text(render_metadata_text(meta), encoding="utf-8")
    return NicheResult(
        script=script,
        narration=narration,
        visuals=visuals,
        metadata=meta,
        short_path=short_path,
        metadata_path=metadata_path,
        out_dir=base,
    )


def run_batch(
    topics: Iterable[str],
    *,
    niche: str = "true_crime",
    out_root: Path | None = None,
) -> list[NicheResult]:
    """Run the niche pipeline over a list of topics. Failures are logged and skipped."""
    results: list[NicheResult] = []
    for topic in topics:
        sub = _subdir(out_root, topic)
        try:
            r = run_niche_pipeline(topic=topic, niche=niche, out_dir=sub)
            results.append(r)
        except Exception as exc:
            if sub is not None:
                shutil.rmtree(sub, ignore_errors=True)
            print(f"[batch] failed topic={topic!r}: {exc}")
    return results


def _subdir(root: Path | None, topic: str) -> Path | None:
    if root is None:
        return None
    return root / f"niche_{slugify(topic)[:40]}"
