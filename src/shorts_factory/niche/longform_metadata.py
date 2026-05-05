"""Long-form video title / description / hashtag generation.

Differs from shorts metadata: the description includes auto-generated
chapter timestamps (so YouTube auto-renders chapters in the player) and
is much longer / more detailed. Hashtags are still <=15.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .broll import BrollAsset
from .longform_assemble import chapter_timestamps_block
from .longform_script import LongformScript
from .longform_voice import LongformNarration
from .script import GeminiNotConfigured, _parse_json_strict


@dataclass(frozen=True)
class LongformMetadata:
    title: str
    description: str
    hashtags: list[str]
    full_description: str  # description + chapters + attribution


_PROMPT = """\
You are writing YouTube metadata for a {duration_min}-minute documentary on
"{topic}" (niche: {niche}).

Return STRICT JSON, no markdown:
{{
  "title": "<70-90 chars, clickable but truthful. NO clickbait emoji spam,
   NO ALL CAPS. Should mention the subject by name.>",
  "description": "<5-8 sentences. First sentence is a hook. Then briefly
   tease the angle. End with one line inviting comments / discussion (NOT
   'subscribe'). Plain text, no markdown.>",
  "hashtags": ["#documentary", ...]   // 8-15 relevant tags
}}

Rules:
- Title under 100 chars. Under 80 is better.
- Description avoids 'In this video', 'subscribe to my channel', and emoji.
- Hashtags lowercase, no spaces, prefix with `#`.

Context:
- Topic: {topic}
- Cold open: {cold_open}
- Chapters:
{chapter_lines}
"""


def _validate_meta_payload(p: dict[str, Any]) -> None:
    for key in ("title", "description", "hashtags"):
        if key not in p:
            raise ValueError(f"Gemini metadata missing required key: {key!r}")
    if not isinstance(p["hashtags"], list):
        raise ValueError("metadata 'hashtags' must be a list")


def _normalise_hashtag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return ""
    if not tag.startswith("#"):
        tag = "#" + tag
    tag = re.sub(r"\s+", "", tag)
    return tag.lower()


def _attribution_block(visuals: list[BrollAsset], script: LongformScript) -> str:
    seen: dict[str, BrollAsset] = {}
    for v in visuals:
        if v.page_url and v.page_url not in seen:
            seen[v.page_url] = v
    lines: list[str] = ["", "— Image credits —"]
    for v in seen.values():
        lines.append(f"• {v.attribution_line}")
    if script.sources:
        lines.append("")
        lines.append("— Sources —")
        for s in script.sources:
            lines.append(f"• {s}")
    return "\n".join(lines)


def generate_longform_metadata(
    script: LongformScript,
    narration: LongformNarration,
    visuals: list[BrollAsset],
    *,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> LongformMetadata:
    """Generate clickable metadata + chapter timestamps + image attribution."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )

    from google import genai
    from google.genai import types

    from .script import _gemini_call_with_retry

    chapter_lines = "\n".join(
        f"  {i}. {ch.title} - {ch.summary}" for i, ch in enumerate(script.chapters, 1)
    )
    prompt = _PROMPT.format(
        topic=script.topic,
        niche=script.niche,
        duration_min=int(round(narration.duration / 60)),
        cold_open=script.cold_open[:200],
        chapter_lines=chapter_lines,
    )

    client = genai.Client(api_key=key)
    resp = _gemini_call_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json_strict(resp.text or "")
    _validate_meta_payload(payload)

    title = str(payload["title"]).strip()
    description = str(payload["description"]).strip()
    hashtags = [_normalise_hashtag(t) for t in payload["hashtags"] if str(t).strip()]
    hashtags = [t for t in hashtags if t and t != "#"]

    chapters_block = chapter_timestamps_block(narration)
    full = (
        description
        + "\n\n"
        + chapters_block
        + "\n\n"
        + " ".join(hashtags)
        + "\n"
        + _attribution_block(visuals, script)
    )
    return LongformMetadata(
        title=title,
        description=description,
        hashtags=hashtags,
        full_description=full,
    )


def render_longform_metadata_text(meta: LongformMetadata) -> str:
    return (
        f"TITLE\n{meta.title}\n\n"
        f"DESCRIPTION\n{meta.full_description}\n\n"
        f"HASHTAGS\n{' '.join(meta.hashtags)}\n"
    )
