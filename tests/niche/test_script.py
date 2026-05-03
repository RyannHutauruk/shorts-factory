"""Tests for niche.script: JSON parsing, validation, file loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shorts_factory.niche.script import (
    GeminiNotConfigured,
    Script,
    ScriptBeat,
    _parse_json_strict,
    _validate_script_payload,
    generate_script,
    script_from_file,
)


def test_parse_json_strict_handles_plain_json() -> None:
    s = '{"a": 1}'
    assert _parse_json_strict(s) == {"a": 1}


def test_parse_json_strict_handles_markdown_fence() -> None:
    s = '```json\n{"a": 1}\n```'
    assert _parse_json_strict(s) == {"a": 1}


def test_parse_json_strict_extracts_object_from_prose() -> None:
    s = 'Here you go: {"a": 1, "b": 2}'
    assert _parse_json_strict(s) == {"a": 1, "b": 2}


def test_validate_script_payload_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="hook"):
        _validate_script_payload({"beats": [], "payoff": ""})


def test_validate_script_payload_rejects_empty_beats() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validate_script_payload({"hook": "", "beats": [], "payoff": ""})


def test_validate_script_payload_rejects_bad_beat() -> None:
    with pytest.raises(ValueError, match="text"):
        _validate_script_payload({"hook": "h", "beats": [{"text": "ok"}], "payoff": "p"})


def test_script_all_text() -> None:
    s = Script(
        topic="x",
        niche="true_crime",
        hook="h.",
        beats=[ScriptBeat(text="b1.", visual_hint="v1"), ScriptBeat(text="b2.", visual_hint="v2")],
        payoff="p.",
    )
    assert s.all_text == "h. b1. b2. p."
    assert s.estimated_seconds > 0


def test_script_from_file(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        "topic": "Hindenburg disaster",
        "hook": "What if luxury became fire in seconds?",
        "beats": [
            {
                "text": "On May 6, 1937, the Hindenburg approached Lakehurst.",
                "visual_hint": "Hindenburg airship 1937",
            },
            {"text": "Fire erupted near the tail.", "visual_hint": "Hindenburg fire 1937"},
        ],
        "payoff": "It ended airship travel.",
        "sources": ["wikipedia.org/wiki/Hindenburg_disaster"],
    }
    p = tmp_path / "script.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    s = script_from_file(p)
    assert s.hook.startswith("What if")
    assert len(s.beats) == 2
    assert s.beats[0].visual_hint == "Hindenburg airship 1937"


def test_generate_script_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiNotConfigured):
        generate_script("topic")


def test_generate_script_calls_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    payload = {
        "hook": "What if the future of air travel vanished in fire?",
        "beats": [
            {
                "text": "The Hindenburg was a symbol of progress.",
                "visual_hint": "Hindenburg airship",
            },
            {"text": "Fire engulfed it on May 6, 1937.", "visual_hint": "Hindenburg fire"},
        ],
        "payoff": "Airship travel ended that day.",
        "sources": ["wikipedia.org/wiki/Hindenburg_disaster"],
    }
    fake_resp = MagicMock()
    fake_resp.text = json.dumps(payload)
    fake_models = MagicMock()
    fake_models.generate_content.return_value = fake_resp
    fake_client = MagicMock()
    fake_client.models = fake_models

    with patch("google.genai.Client", return_value=fake_client):
        s = generate_script("Hindenburg disaster")

    assert s.hook.startswith("What if")
    assert len(s.beats) == 2
    assert s.payoff.startswith("Airship")
