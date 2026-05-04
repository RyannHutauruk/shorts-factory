"""Generate a chaptered, long-form (~5-25 min) explainer script via Gemini.

The shorts pipeline produces a single ~120-word punchy script. Long-form
documentaries on YouTube need a different shape: cold-open hook, 4-7
chapters with their own internal arcs, and a closing payoff. We split
generation into two LLM calls so each step stays focused:

  1. ``generate_outline`` -> chapter titles + 1-line summaries + b-roll keywords
  2. ``expand_chapter`` -> per-chapter prose paragraphs with visual hints

Combined, this produces a ``LongformScript`` whose ``all_text`` reads as
a complete documentary script.

Defaults assume ``2.6 wps`` documentary narration (matches Piper Ryan).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .script import (
    GeminiNotConfigured,
    _gemini_call_with_retry,
    _parse_json_strict,
)

# Delay between consecutive chapter-expansion calls to stay under the
# Gemini free-tier per-minute quota (15 req/min for flash-lite). 5s between
# 6 chapter calls = ~25s total, well clear of the 60s window.
_INTER_CHAPTER_SLEEP_S = 5.0

WORDS_PER_SECOND = 2.6


@dataclass(frozen=True)
class LongformBeat:
    """A single narration paragraph within a chapter."""

    text: str
    visual_hint: str


@dataclass(frozen=True)
class LongformChapter:
    """A chapter: title, transition line, beats, optional outro line."""

    index: int  # 1-based chapter number
    title: str
    summary: str  # 1-line description (used in outline + chapter timestamps)
    transition: str  # 1-2 sentence intro that leads into the chapter
    beats: list[LongformBeat]
    closing: str  # 1 sentence wrap that hands off to next chapter

    @property
    def all_text(self) -> str:
        parts = [self.transition] + [b.text for b in self.beats] + [self.closing]
        return " ".join(p.strip() for p in parts if p)

    @property
    def word_count(self) -> int:
        return len(self.all_text.split())

    @property
    def estimated_seconds(self) -> float:
        return self.word_count / WORDS_PER_SECOND


@dataclass(frozen=True)
class LongformScript:
    """Full long-form documentary script."""

    topic: str
    niche: str
    cold_open: str  # first 20-40 sec hook before chapter 1
    chapters: list[LongformChapter]
    outro: str  # closing 20-40 sec, no 'subscribe' filler
    sources: list[str] = field(default_factory=list)
    duration_target_min: float = 10.0

    @property
    def all_text(self) -> str:
        chunks: list[str] = [self.cold_open]
        for ch in self.chapters:
            chunks.append(ch.all_text)
        chunks.append(self.outro)
        return " ".join(c.strip() for c in chunks if c)

    @property
    def word_count(self) -> int:
        return len(self.all_text.split())

    @property
    def estimated_seconds(self) -> float:
        return self.word_count / WORDS_PER_SECOND

    @property
    def chapter_timestamps(self) -> list[tuple[int, str]]:
        """Cumulative chapter start times (seconds) + chapter title.

        Includes a leading ``(0, "Cold open")`` and timestamps for each
        chapter relative to the cold-open length. Suitable for embedding
        in a YouTube description so the player auto-shows chapters.
        """
        result: list[tuple[int, str]] = [(0, "Cold open")]
        cursor = len(self.cold_open.split()) / WORDS_PER_SECOND
        for ch in self.chapters:
            result.append((int(cursor), ch.title))
            cursor += ch.estimated_seconds
        return result


# ---------- Prompt templates ----------

_OUTLINE_PROMPT = """You are designing a YOUTUBE LONG-FORM DOCUMENTARY SHORT
in the '{niche}' niche on the topic: "{topic}".

Target total length: {duration_min} minutes of spoken narration
(~{total_words} words at 2.6 words/sec, English, US audience).

Your job is to PLAN the documentary - not write it yet.

Return STRICT JSON, no markdown fences, no commentary. Schema:

{{
  "cold_open_summary": "<one paragraph (~80 words) describing the opening hook
   that runs before chapter 1. Establishes stakes, asks a compelling question,
   or drops a striking fact. No 'In this video...'>",
  "chapters": [
    {{
      "title": "<short chapter title, 2-6 words, no numbering>",
      "summary": "<one sentence describing what this chapter covers>",
      "key_points": ["<3-5 bullet points the chapter must hit>", "..."],
      "broll_keywords": ["<concrete real-world noun phrases for image search>",
                         "<5-8 keywords>"]
    }},
    ... {chapter_count} chapters total ...
  ],
  "outro_summary": "<one paragraph (~70 words) describing the closing payoff.
   Lands the lesson or final twist. No 'subscribe'.>",
  "sources": ["wikipedia.org/wiki/<page>", ...]
}}

Rules:
- {chapter_count} chapters that flow as a real narrative arc
  (setup -> escalation -> turning point -> consequence -> reflection).
- Every chapter title is a proper, search-friendly phrase a viewer would
  recognise as a section heading. NOT clickbait, NOT a question.
- broll_keywords MUST be concrete searchable noun phrases (people, places,
  objects, events) - not abstract concepts.
- Facts must be accurate. If a number or detail is uncertain, omit it.
- DO NOT include any commentary, headings, or markdown - JSON only.

{audience_bias}
"""

_CHAPTER_EXPANSION_PROMPT = """You are writing CHAPTER {chapter_index} of a
YouTube long-form documentary on the topic: "{topic}".

The chapter is titled: "{chapter_title}"
Its job: {chapter_summary}
Key points it must hit:
{key_points_bulleted}

Style:
- {niche_style}
- Calm, factual, documentary tone. No tabloid energy. No 'In this video'.
- US English, US audience. Lead with concrete details over abstractions.

Length: ~{words_target} words of spoken narration (~{seconds_target} seconds at
2.6 wps).

Return STRICT JSON, no markdown, no commentary. Schema:

{{
  "transition": "<1-2 sentences that bridge from the previous chapter into
   this one. The very first chapter's transition reads as a chapter opener,
   not a callback.>",
  "beats": [
    {{
      "text": "<one paragraph of narration, 2-5 sentences, that advances
       the chapter's arc>",
      "visual_hint": "<concrete searchable noun phrase, e.g. 'space shuttle
       Challenger launch 1986'>"
    }},
    ... 4-7 beats total ...
  ],
  "closing": "<1 sentence that lands this chapter and hands off to the next
   without naming chapter numbers>"
}}

Rules:
- The cumulative narration MUST land near {words_target} words (+/- 15%).
- visual_hint must be a real searchable noun phrase, not a metaphor.
- No emojis. No hashtags. No CTAs.
"""

_NICHE_STYLES: dict[str, str] = {
    "true_crime": "Calm, tense, dramatic without sensationalism. Specific names "
    "and dates. Empathetic toward victims, never lurid.",
    "history": "Authoritative, curious, fact-grounded. Cites primary sources "
    "where natural. Avoids 'lost civilization' clickbait.",
    "science": "Punchy, accessible, accurate. Concrete examples over jargon. "
    "Cites missions / studies / observations.",
    "mysteries": "Curious, level-headed. Distinguishes verified facts from "
    "popular speculation. Avoids supernatural claims.",
    "weird_facts": "Wry, surprising, but factually grounded. Cites the "
    "specific instance, not 'scientists say'.",
    "biographies": "Empathetic, factual, no hagiography. Origin -> turning "
    "point -> legacy. Public figures only.",
    "tech_history": "Wry, fond, precise. Names specific gadgets, formats, "
    "people. Why it failed AND why it mattered.",
    "space": "Awe-struck but grounded. Cites missions, observations, "
    "telescopes. Distinguishes hypothesis from established result.",
}


_AUDIENCE_BIAS: dict[str, str] = {
    "US": "Audience: US viewers. Bias toward subjects with US recognition "
    "(US history/people, North American mysteries, NASA missions, US tech "
    "history). Globally famous topics are fine; just avoid niche regional "
    "content with no US recognition.",
    "UK": "Audience: UK / Commonwealth viewers. Bias toward UK history, "
    "people, places, Commonwealth-relevant subjects.",
    "global": "Audience: global English-speaking viewers. No regional bias.",
    "general": "General audience. No regional bias.",
}


# ---------- Public API ----------


def _resolve_chapter_count(duration_min: float) -> int:
    """Heuristic: 1 chapter per ~2 minutes, clamped to 4-7."""
    return max(4, min(7, round(duration_min / 2.0)))


def _resolve_total_words(duration_min: float) -> int:
    return int(round(duration_min * 60 * WORDS_PER_SECOND))


def _gemini_client(api_key: str | None) -> Any:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )
    from google import genai

    return genai.Client(api_key=key)


def generate_outline(
    topic: str,
    *,
    niche: str = "history",
    duration_min: float = 10.0,
    audience: str = "US",
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Plan the documentary: returns the raw outline JSON dict.

    Used as a stepping stone to ``expand_chapter``; the caller normally
    uses ``generate_longform_script`` which orchestrates both.
    """
    if niche not in _NICHE_STYLES:
        raise ValueError(f"unknown niche {niche!r}; choose from {sorted(_NICHE_STYLES)}")
    if audience not in _AUDIENCE_BIAS:
        raise ValueError(f"unknown audience {audience!r}; choose from {sorted(_AUDIENCE_BIAS)}")

    chapter_count = _resolve_chapter_count(duration_min)
    total_words = _resolve_total_words(duration_min)
    prompt = _OUTLINE_PROMPT.format(
        topic=topic,
        niche=niche,
        duration_min=int(duration_min),
        total_words=total_words,
        chapter_count=chapter_count,
        audience_bias=_AUDIENCE_BIAS[audience],
    )

    from google.genai import types

    client = _gemini_client(api_key)
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
    _validate_outline_payload(payload, chapter_count)
    return payload


def expand_chapter(
    *,
    topic: str,
    niche: str,
    chapter_index: int,
    chapter_title: str,
    chapter_summary: str,
    key_points: list[str],
    duration_min: float = 10.0,
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> LongformChapter:
    """Expand a single chapter outline into prose narration."""
    chapter_count = _resolve_chapter_count(duration_min)
    total_words = _resolve_total_words(duration_min)
    # Reserve cold open + outro at ~150 words combined; split the rest evenly.
    body_words = max(400, total_words - 150)
    words_target = max(180, body_words // chapter_count)
    seconds_target = int(round(words_target / WORDS_PER_SECOND))

    points_bulleted = "\n".join(f"  - {p}" for p in key_points) or "  - (none)"
    prompt = _CHAPTER_EXPANSION_PROMPT.format(
        topic=topic,
        chapter_index=chapter_index,
        chapter_title=chapter_title,
        chapter_summary=chapter_summary,
        key_points_bulleted=points_bulleted,
        niche_style=_NICHE_STYLES[niche],
        words_target=words_target,
        seconds_target=seconds_target,
    )

    from google.genai import types

    client = _gemini_client(api_key)
    resp = _gemini_call_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.65,
            response_mime_type="application/json",
        ),
    )
    payload = _parse_json_strict(resp.text or "")
    _validate_chapter_payload(payload)

    beats = [
        LongformBeat(
            text=str(b["text"]).strip(),
            visual_hint=str(b["visual_hint"]).strip(),
        )
        for b in payload["beats"]
    ]
    return LongformChapter(
        index=chapter_index,
        title=chapter_title,
        summary=chapter_summary,
        transition=str(payload["transition"]).strip(),
        beats=beats,
        closing=str(payload["closing"]).strip(),
    )


def generate_longform_script(
    topic: str,
    *,
    niche: str = "history",
    duration_min: float = 10.0,
    audience: str = "US",
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> LongformScript:
    """End-to-end: outline + per-chapter expansion -> ``LongformScript``."""
    outline = generate_outline(
        topic,
        niche=niche,
        duration_min=duration_min,
        audience=audience,
        model=model,
        api_key=api_key,
    )

    chapters: list[LongformChapter] = []
    for i, ch_outline in enumerate(outline["chapters"], start=1):
        if i > 1:
            time.sleep(_INTER_CHAPTER_SLEEP_S)
        chapter = expand_chapter(
            topic=topic,
            niche=niche,
            chapter_index=i,
            chapter_title=str(ch_outline["title"]).strip(),
            chapter_summary=str(ch_outline["summary"]).strip(),
            key_points=[str(k) for k in ch_outline.get("key_points", [])],
            duration_min=duration_min,
            model=model,
            api_key=api_key,
        )
        chapters.append(chapter)

    return LongformScript(
        topic=topic,
        niche=niche,
        cold_open=str(outline["cold_open_summary"]).strip(),
        chapters=chapters,
        outro=str(outline["outro_summary"]).strip(),
        sources=[str(s).strip() for s in outline.get("sources", [])],
        duration_target_min=duration_min,
    )


# ---------- Validation + IO ----------


def _validate_outline_payload(payload: dict[str, Any], chapter_count: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError("outline JSON must be an object")
    for key in ("cold_open_summary", "chapters", "outro_summary"):
        if key not in payload:
            raise ValueError(f"outline missing key {key!r}")
    if not isinstance(payload["chapters"], list) or len(payload["chapters"]) == 0:
        raise ValueError("outline 'chapters' must be a non-empty list")
    # Allow Gemini some flex on count, but reject wildly off responses.
    n = len(payload["chapters"])
    if not (chapter_count - 2 <= n <= chapter_count + 2):
        raise ValueError(
            f"outline returned {n} chapters; expected ~{chapter_count} "
            f"({chapter_count - 2}-{chapter_count + 2})"
        )
    for i, ch in enumerate(payload["chapters"]):
        if not isinstance(ch, dict):
            raise ValueError(f"chapter[{i}] is not an object")
        for key in ("title", "summary"):
            if key not in ch:
                raise ValueError(f"chapter[{i}] missing key {key!r}")


def _validate_chapter_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("chapter JSON must be an object")
    for key in ("transition", "beats", "closing"):
        if key not in payload:
            raise ValueError(f"chapter missing key {key!r}")
    if not isinstance(payload["beats"], list) or not payload["beats"]:
        raise ValueError("chapter 'beats' must be a non-empty list")
    for i, b in enumerate(payload["beats"]):
        if not isinstance(b, dict):
            raise ValueError(f"chapter beat[{i}] is not an object")
        if "text" not in b or "visual_hint" not in b:
            raise ValueError(f"chapter beat[{i}] missing 'text' or 'visual_hint'")


def script_to_dict(script: LongformScript) -> dict[str, Any]:
    """Serialise a ``LongformScript`` to a JSON-friendly dict."""
    return {
        "topic": script.topic,
        "niche": script.niche,
        "duration_target_min": script.duration_target_min,
        "cold_open": script.cold_open,
        "chapters": [
            {
                "index": ch.index,
                "title": ch.title,
                "summary": ch.summary,
                "transition": ch.transition,
                "beats": [{"text": b.text, "visual_hint": b.visual_hint} for b in ch.beats],
                "closing": ch.closing,
            }
            for ch in script.chapters
        ],
        "outro": script.outro,
        "sources": script.sources,
    }


def script_from_dict(payload: dict[str, Any]) -> LongformScript:
    """Inverse of ``script_to_dict``: load a script from a saved JSON dict."""
    chapters = [
        LongformChapter(
            index=int(ch["index"]),
            title=str(ch["title"]),
            summary=str(ch.get("summary", "")),
            transition=str(ch["transition"]),
            beats=[
                LongformBeat(text=str(b["text"]), visual_hint=str(b["visual_hint"]))
                for b in ch["beats"]
            ],
            closing=str(ch["closing"]),
        )
        for ch in payload["chapters"]
    ]
    return LongformScript(
        topic=str(payload["topic"]),
        niche=str(payload["niche"]),
        cold_open=str(payload["cold_open"]),
        chapters=chapters,
        outro=str(payload["outro"]),
        sources=[str(s) for s in payload.get("sources", [])],
        duration_target_min=float(payload.get("duration_target_min", 10.0)),
    )


def script_from_file(path: str | os.PathLike[str]) -> LongformScript:
    """Load a hand-written long-form script from a JSON file (no LLM needed)."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return script_from_dict(payload)
