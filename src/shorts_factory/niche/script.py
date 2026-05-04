"""Generate a short-form explainer script from a topic using Google Gemini.

Defaults to ``true_crime`` niche style: dramatic, fact-grounded, hook in
first 3 seconds. Other niches can be added later via the ``NICHE_PROMPTS``
mapping.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# google-genai is the v2 SDK; lazy import in functions so unit tests can patch
# the module without a hard dependency.

GeminiNotConfigured = RuntimeError


@dataclass(frozen=True)
class ScriptBeat:
    """A single sentence/shot in the body of the script."""

    text: str
    visual_hint: str  # short query string for B-roll search (e.g. "Hindenburg airship explosion")


@dataclass(frozen=True)
class Script:
    """Structured 30-50 second narration script."""

    topic: str
    niche: str
    hook: str
    beats: list[ScriptBeat]
    payoff: str
    sources: list[str] = field(default_factory=list)  # short URLs for fact attribution

    @property
    def all_text(self) -> str:
        chunks = [self.hook] + [b.text for b in self.beats] + [self.payoff]
        return " ".join(c.strip() for c in chunks if c)

    @property
    def estimated_seconds(self) -> float:
        # Rough: 2.6 words/sec for documentary narration.
        word_count = len(self.all_text.split())
        return word_count / 2.6


# A short, opinionated prompt template per niche. The model is asked to
# return strict JSON; we then parse and validate.
NICHE_PROMPTS: dict[str, str] = {
    "true_crime": (
        "You are writing a 35-45 second YouTube Short script in the 'true crime / "
        "disasters' niche. The narrator is calm but tense, dramatic without being "
        "tabloid. Facts must be accurate; if you're unsure of a number, omit it.\n"
        "\n"
        "Format STRICTLY as JSON, no markdown, no commentary. Schema:\n"
        "{\n"
        '  "hook": "1 sentence, <=15 words, must grab attention in 3 seconds. '
        'Often a question or a striking fact.",\n'
        '  "beats": [\n'
        '    {"text": "1-2 sentence beat.", "visual_hint": "concrete noun phrase '
        "for image search, e.g. 'Hindenburg airship docking 1937'\"},\n"
        "    ... 3 to 5 beats total ...\n"
        "  ],\n"
        '  "payoff": "1 closing line that lands the lesson or final twist.",\n'
        '  "sources": ["wikipedia.org/wiki/<page>", ...]\n'
        "}\n"
        "\n"
        "Rules:\n"
        "- Total spoken length 90-130 words (35-45 sec at 2.6 wps)\n"
        "- No 'In this video...' filler. No 'subscribe'. No emojis. No hashtags here.\n"
        "- The hook must NOT start with 'In <year>' or a date - lead with stakes.\n"
        "- visual_hint must be a real-world searchable noun phrase, not a metaphor.\n"
    ),
    "history": (
        "You are writing a 35-45 second YouTube Short script in the 'history / "
        "lost civilizations' niche. Authoritative, curious, fact-grounded.\n"
        "Format STRICTLY as JSON, no markdown. Schema:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words, must intrigue immediately.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ... 3-5 ...],\n'
        '  "payoff": "1 closing line.",\n'
        '  "sources": ["wikipedia.org/wiki/...", ...]\n'
        "}\n"
        "Total 90-130 words. No filler. visual_hint must be a real noun phrase.\n"
    ),
    "science": (
        "You are writing a 35-45 second YouTube Short in the 'science explainer' "
        "niche. Punchy, accessible, accurate. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ...3-5...],\n'
        '  "payoff": "1 line.",\n'
        '  "sources": [...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
    "mysteries": (
        "You are writing a 35-45 second YouTube Short in the 'unsolved mysteries' "
        "niche: disappearances, unexplained phenomena, ancient puzzles. Curious "
        "and grounded - never invent 'facts'. If a theory is speculative, mark "
        "it with 'one theory is...'. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words, often a question.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ...3-5...],\n'
        '  "payoff": "1 line - what remains unknown.",\n'
        '  "sources": ["wikipedia.org/wiki/...", ...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
    "weird_facts": (
        "You are writing a 35-45 second YouTube Short of weird-but-true facts on "
        "a single theme. Tone: amused, surprising, never sensational. Every fact "
        "must be verifiable; omit anything you'd hedge on. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words.",\n'
        '  "beats": [{"text": "1 fact in 1-2 sentences.", "visual_hint": "..."}, ...3-5...],\n'
        '  "payoff": "1 closing line that ties them together.",\n'
        '  "sources": ["wikipedia.org/wiki/...", ...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
    "biographies": (
        "You are writing a 35-45 second YouTube Short profile of a fascinating "
        "person (historical or modern, but only public figures). Empathetic, "
        "factual, no hagiography. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words. Lead with stakes or paradox.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ...3-5 covering origin, turning point, legacy...],\n'
        '  "payoff": "1 line.",\n'
        '  "sources": ["wikipedia.org/wiki/...", ...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
    "tech_history": (
        "You are writing a 35-45 second YouTube Short in the 'forgotten tech / "
        "failed inventions' niche - gadgets, formats, ideas that flopped or "
        "vanished. Wry, fond, accurate. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ...3-5...],\n'
        '  "payoff": "1 line - why it matters today.",\n'
        '  "sources": ["wikipedia.org/wiki/...", ...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
    "space": (
        "You are writing a 35-45 second YouTube Short in the 'space / cosmos' "
        "niche - missions, exoplanets, black holes, weird physics. Awe-struck "
        "but grounded; cite missions/observations, not vibes. JSON only:\n"
        "{\n"
        '  "hook": "1 sentence <=15 words.",\n'
        '  "beats": [{"text": "...", "visual_hint": "..."}, ...3-5...],\n'
        '  "payoff": "1 line.",\n'
        '  "sources": ["wikipedia.org/wiki/...", "nasa.gov/...", ...]\n'
        "}\n"
        "Total 90-130 words. visual_hint must be a real searchable noun phrase.\n"
    ),
}


def _parse_json_strict(raw: str) -> dict[str, Any]:
    """Best-effort JSON parse: strip markdown code fences if the model added any."""
    s = raw.strip()
    # Strip ```json fences.
    fence = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1)
    # Some models prepend a sentence before the JSON; salvage by extracting first {...}.
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    return json.loads(s)


def _validate_script_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Gemini did not return a JSON object.")
    for key in ("hook", "beats", "payoff"):
        if key not in payload:
            raise ValueError(f"Gemini script missing required key: {key!r}")
    if not isinstance(payload["beats"], list) or not payload["beats"]:
        raise ValueError("Gemini script 'beats' must be a non-empty list.")
    for i, beat in enumerate(payload["beats"]):
        if not isinstance(beat, dict):
            raise ValueError(f"beat[{i}] is not a JSON object")
        if "text" not in beat or "visual_hint" not in beat:
            raise ValueError(f"beat[{i}] missing 'text' or 'visual_hint'")


def generate_script(
    topic: str,
    *,
    niche: str = "true_crime",
    model: str = "gemini-2.5-flash-lite",
    api_key: str | None = None,
) -> Script:
    """Generate a structured ``Script`` for ``topic`` using Gemini.

    The Gemini SDK is invoked via ``google.genai.Client``. ``GEMINI_API_KEY``
    is read from the environment by the SDK; we pass it explicitly when given.
    On free-tier ``RESOURCE_EXHAUSTED`` 429s we honour the API's suggested
    retry delay and retry up to 3 times.
    """
    if niche not in NICHE_PROMPTS:
        raise ValueError(f"unknown niche {niche!r}; choose from {sorted(NICHE_PROMPTS)}")

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise GeminiNotConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/apikey"
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    prompt = NICHE_PROMPTS[niche] + f"\nTopic: {topic}\n"
    response = _gemini_call_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    raw = response.text or ""
    payload = _parse_json_strict(raw)
    _validate_script_payload(payload)

    return Script(
        topic=topic,
        niche=niche,
        hook=str(payload["hook"]).strip(),
        beats=[
            ScriptBeat(
                text=str(b["text"]).strip(),
                visual_hint=str(b["visual_hint"]).strip(),
            )
            for b in payload["beats"]
        ],
        payoff=str(payload["payoff"]).strip(),
        sources=[str(s).strip() for s in (payload.get("sources") or [])],
    )


_RATE_LIMIT_MARKERS = ("RESOURCE_EXHAUSTED", "429")
_TRANSIENT_MARKERS = ("UNAVAILABLE", "503", "INTERNAL", "500", "DEADLINE_EXCEEDED", "504")


def _gemini_call_with_retry(
    client: Any,
    *,
    model: str,
    contents: str,
    config: Any,
    max_attempts: int = 6,
) -> Any:
    """Call ``client.models.generate_content`` with retry on transient errors.

    Retries on:
      - 429 ``RESOURCE_EXHAUSTED`` (free-tier rate limit). Honours the
        ``retry in Ns`` hint when the API includes one.
      - 503 ``UNAVAILABLE`` / 500 ``INTERNAL`` / 504 ``DEADLINE_EXCEEDED``
        (server-side overload). Uses exponential backoff: 5, 15, 30, 60s.

    All other errors are surfaced immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:  # noqa: BLE001 - genai raises ClientError subclass
            last_exc = exc
            msg = str(exc)
            is_rate_limit = any(marker in msg for marker in _RATE_LIMIT_MARKERS)
            is_transient = any(marker in msg for marker in _TRANSIENT_MARKERS)

            if not (is_rate_limit or is_transient):
                raise

            if is_rate_limit:
                wait_s = 30.0
                m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
                if m:
                    wait_s = float(m.group(1)) + 2.0
                wait_s = max(wait_s, 5.0)
                label = "429 throttled"
            else:
                # exponential backoff: 5s, 15s, 30s, 60s
                wait_s = min(5.0 * (3**attempt), 60.0)
                label = "503/transient"

            if attempt + 1 >= max_attempts:
                break
            print(
                f"[gemini] {label}, sleeping {wait_s:.1f}s (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


def script_from_file(path: str | os.PathLike[str]) -> Script:
    """Load a hand-written script (no LLM) from a JSON file with the same schema."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    _validate_script_payload(payload)
    return Script(
        topic=str(payload.get("topic", "")),
        niche=str(payload.get("niche", "manual")),
        hook=str(payload["hook"]).strip(),
        beats=[
            ScriptBeat(text=str(b["text"]).strip(), visual_hint=str(b["visual_hint"]).strip())
            for b in payload["beats"]
        ],
        payoff=str(payload["payoff"]).strip(),
        sources=[str(s).strip() for s in (payload.get("sources") or [])],
    )
