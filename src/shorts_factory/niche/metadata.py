"""Generate clickable title / description / hashtags via Gemini.

The description is automatically suffixed with B-roll source attribution so
the channel stays compliant with Wikimedia Commons reuse terms.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from .broll import BrollAsset
from .script import GeminiNotConfigured, Script, _parse_json_strict


@dataclass(frozen=True)
class ShortMetadata:
    title: str  # <=60 char clickable title
    description: str  # multi-line description body
    hashtags: list[str]
    full_description: str  # description + attribution block, ready to paste


METADATA_PROMPT = """\
You are writing the YouTube Shorts metadata for a 35-45 second narrated explainer.

Return STRICT JSON, no markdown:
{{
  "title": "<= 60 chars, clickable but truthful, no clickbait emoji spam, no ALL CAPS",
  "description": "2-4 sentences. Hook the viewer + tease the answer. End with a call-to-engage line.",
  "hashtags": ["#truecrime", "#disasters", "#history", ...]   // 5-10 relevant tags
}}

Rules:
- Title under 60 chars; under 50 is better. No 'YOU WON'T BELIEVE'.
- Description avoids 'In this video' filler. Avoid 'subscribe' (YouTube discourages bare subscribe begs).
- Hashtags lowercase, no spaces, prefix with `#`.
- Topic: {topic}
- Hook line from the script: {hook}
- Payoff line: {payoff}
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


def _attribution_block(visuals: list[BrollAsset], script: Script) -> str:
    seen: dict[str, BrollAsset] = {}
    for v in visuals:
        if v.page_url and v.page_url not in seen:
            seen[v.page_url] = v
    lines = ["", "— Image credits —"]
    for v in seen.values():
        lines.append(f"• {v.attribution_line}")
    if script.sources:
        lines.append("")
        lines.append("— Sources —")
        for s in script.sources:
            lines.append(f"• {s}")
    return "\n".join(lines)


def generate_metadata(
    script: Script,
    visuals: list[BrollAsset],
    *,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> ShortMetadata:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )
    from google import genai
    from google.genai import types

    from .script import _gemini_call_with_retry

    client = genai.Client(api_key=key)
    prompt = METADATA_PROMPT.format(topic=script.topic, hook=script.hook, payoff=script.payoff)
    resp = _gemini_call_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.75, response_mime_type="application/json"),
    )
    payload = _parse_json_strict(resp.text or "")
    _validate_meta_payload(payload)

    title = str(payload["title"]).strip()
    description = str(payload["description"]).strip()
    hashtags = [_normalise_hashtag(t) for t in payload["hashtags"] if str(t).strip()]
    hashtags = [t for t in hashtags if t and t != "#"]

    full = description + "\n\n" + " ".join(hashtags) + "\n" + _attribution_block(visuals, script)
    return ShortMetadata(
        title=title,
        description=description,
        hashtags=hashtags,
        full_description=full,
    )


def render_metadata_text(meta: ShortMetadata) -> str:
    """Plain text dump - what gets written to METADATA.txt next to the MP4."""
    return (
        f"TITLE\n{meta.title}\n\n"
        f"DESCRIPTION\n{meta.full_description}\n\n"
        f"HASHTAGS\n{' '.join(meta.hashtags)}\n"
    )


__all__ = ["ShortMetadata", "generate_metadata", "render_metadata_text"]
