"""Tests for niche.topics: Gemini topic discovery payload validation + dedup."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shorts_factory.niche.topics import (
    TopicIdea,
    _validate_topics_payload,
    discover_topics,
    render_topics_text,
)


def test_validate_topics_dedups_case_insensitively() -> None:
    payload = {
        "topics": [
            {"topic": "Hindenburg disaster", "angle": "a", "why_interesting": "b"},
            {"topic": "  hindenburg disaster  ", "angle": "x", "why_interesting": "y"},
            {"topic": "Tenerife airport disaster"},
        ]
    }
    out = _validate_topics_payload(payload, expected=10)
    assert len(out) == 2
    titles = {x["topic"].lower().strip() for x in out}
    assert titles == {"hindenburg disaster", "tenerife airport disaster"}


def test_validate_topics_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        _validate_topics_payload({"topics": []}, expected=5)


def test_validate_topics_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        _validate_topics_payload(["foo"], expected=5)


def test_validate_topics_skips_unusable_items() -> None:
    payload = {"topics": [{"angle": "no topic key"}, {"topic": "Halifax explosion"}]}
    out = _validate_topics_payload(payload, expected=10)
    assert len(out) == 1
    assert out[0]["topic"] == "Halifax explosion"


def test_render_topics_text_includes_angles_and_why() -> None:
    topics = [
        TopicIdea(topic="A", angle="ang", why_interesting="why"),
        TopicIdea(topic="B", angle="", why_interesting=""),
    ]
    text = render_topics_text(topics)
    assert "A\n" in text
    assert "# angle: ang" in text
    assert "# why: why" in text
    assert "B\n" in text


def _fake_gemini_response(items: list[dict[str, Any]]) -> Any:
    resp = MagicMock()
    resp.text = json.dumps({"topics": items})
    return resp


def test_discover_topics_calls_gemini_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [{"topic": f"Topic {i}", "angle": "a", "why_interesting": "w"} for i in range(5)]

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_gemini_response(items)

    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch.dict(
        "sys.modules", {"google": MagicMock(genai=fake_genai), "google.genai": fake_genai}
    ):
        with patch("shorts_factory.niche.script._gemini_call_with_retry") as call:
            call.return_value = _fake_gemini_response(items)
            out = discover_topics(niche="history", count=5)

    assert len(out) == 5
    assert all(isinstance(t, TopicIdea) for t in out)
    assert out[0].topic == "Topic 0"


def test_discover_topics_rejects_unknown_niche() -> None:
    with pytest.raises(ValueError):
        discover_topics(niche="not-a-niche", count=5, api_key="fake")


def test_discover_topics_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        discover_topics(niche="history", count=3)


def test_discover_topics_threads_audience_bias_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'audience' arg should be reflected in the rendered Gemini prompt."""
    items = [{"topic": "Topic", "angle": "a", "why_interesting": "w"}]

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_gemini_response(items)
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch.dict(
        "sys.modules", {"google": MagicMock(genai=fake_genai), "google.genai": fake_genai}
    ):
        discover_topics(niche="history", count=1, audience="US")

    assert fake_client.models.generate_content.call_count == 1
    sent_prompt = fake_client.models.generate_content.call_args.kwargs["contents"]
    assert "United States" in sent_prompt
    assert "US audience" in sent_prompt
