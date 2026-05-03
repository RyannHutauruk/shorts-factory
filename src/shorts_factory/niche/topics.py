"""Topic discovery: ask Gemini for N fresh, verifiable topic ideas in a niche.

Used to seed the batch generator and the local scheduler so the channel
gets a steady stream of diverse, evergreen-but-fresh topics. We instruct
the model to favour topics that have a Wikipedia page (so B-roll search
on Wikimedia Commons has a fighting chance).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .script import (
    NICHE_PROMPTS,
    GeminiNotConfigured,
    _gemini_call_with_retry,
    _parse_json_strict,
)

DEFAULT_TOPIC_COUNT = 30


@dataclass(frozen=True)
class TopicIdea:
    """A single suggested topic with optional context for the script writer."""

    topic: str
    angle: str  # the specific framing, e.g. "the role of fog vs miscommunication"
    why_interesting: str  # one-line rationale; surfaced to humans only


_TOPIC_PROMPT = """You are a research producer for a faceless YouTube Shorts channel.
The channel niche is: {niche_label}.

Generate exactly {count} distinct topic ideas suitable for a 35-50 second
explainer short. Rules:
- Each topic MUST have a Wikipedia article (so we can search Wikimedia Commons
  for B-roll).
- No two topics should overlap in subject. Mix well-known with lesser-known
  but verifiable subjects.
- No clickbait fabrications. No conspiracy theories.
- Avoid: events still legally pending; living individuals' private lives; recent
  unverified breaking news.
- Topics should be specific enough that a single 45s short can do them justice
  ("Tenerife airport disaster" GOOD; "aviation history" BAD).
- For 'biographies' niche, only public figures with substantial public records.
- Avoid topics already in this exclusion list: {avoid_list}

Output STRICT JSON, no markdown:
{{
  "topics": [
    {{
      "topic": "<short canonical title, search-friendly>",
      "angle": "<one-line specific framing>",
      "why_interesting": "<one-line rationale>"
    }},
    ... {count} total ...
  ]
}}
"""

_NICHE_LABELS: dict[str, str] = {
    "true_crime": "true crime / disasters / historical catastrophes",
    "history": "history / lost civilizations / pivotal events",
    "science": "science explainers / physics / biology / how things work",
    "mysteries": "unsolved mysteries / disappearances / unexplained phenomena",
    "weird_facts": "weird-but-verifiable facts / strange biology / odd laws",
    "biographies": "biographies of fascinating public figures",
    "tech_history": "forgotten tech / failed inventions / obsolete formats",
    "space": "space / cosmos / missions / exoplanets / weird physics",
}


def _validate_topics_payload(payload: Any, *, expected: int) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("Gemini topic discovery did not return a JSON object")
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("Gemini topic discovery missing 'topics' list")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        norm = re.sub(r"\s+", " ", topic.lower())
        if norm in seen:
            continue
        seen.add(norm)
        out.append(
            {
                "topic": topic,
                "angle": str(item.get("angle", "")).strip(),
                "why_interesting": str(item.get("why_interesting", "")).strip(),
            }
        )
    if not out:
        raise ValueError("Gemini topic discovery produced no usable topics")
    # Don't error if we got fewer than expected - the model sometimes trims.
    return out[:expected]


def discover_topics(
    *,
    niche: str = "history",
    count: int = DEFAULT_TOPIC_COUNT,
    avoid: list[str] | None = None,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> list[TopicIdea]:
    """Ask Gemini for ``count`` fresh topic ideas in ``niche``."""
    if niche not in NICHE_PROMPTS:
        raise ValueError(f"unknown niche {niche!r}; choose from {sorted(NICHE_PROMPTS)}")
    label = _NICHE_LABELS.get(niche, niche.replace("_", " "))

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    avoid_list = json.dumps(avoid or [])
    prompt = _TOPIC_PROMPT.format(niche_label=label, count=count, avoid_list=avoid_list)
    resp = _gemini_call_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.9,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json_strict(resp.text or "")
    items = _validate_topics_payload(payload, expected=count)
    return [
        TopicIdea(
            topic=i["topic"],
            angle=i.get("angle", ""),
            why_interesting=i.get("why_interesting", ""),
        )
        for i in items
    ]


def render_topics_text(topics: list[TopicIdea]) -> str:
    """Plain text dump - what gets written to the topic queue file."""
    lines = []
    for t in topics:
        lines.append(t.topic)
        if t.angle:
            lines.append(f"  # angle: {t.angle}")
        if t.why_interesting:
            lines.append(f"  # why: {t.why_interesting}")
    return "\n".join(lines) + "\n"


__all__ = ["TopicIdea", "discover_topics", "render_topics_text"]
