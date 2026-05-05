"""Tests for niche.metadata: hashtag normalization, attribution block."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_factory.niche.broll import BrollAsset
from shorts_factory.niche.metadata import (
    ShortMetadata,
    _attribution_block,
    _normalise_hashtag,
    generate_metadata,
    render_metadata_text,
)
from shorts_factory.niche.script import Script, ScriptBeat


def _sample_script() -> Script:
    return Script(
        topic="Hindenburg disaster",
        niche="true_crime",
        hook="What if luxury became fire?",
        beats=[
            ScriptBeat(text="The Hindenburg was a marvel.", visual_hint="Hindenburg airship"),
        ],
        payoff="Airship travel ended that day.",
        sources=["wikipedia.org/wiki/Hindenburg_disaster"],
    )


def _sample_asset() -> BrollAsset:
    return BrollAsset(
        query="Hindenburg",
        title="Hindenburg disaster.jpg",
        local_path=Path("/tmp/h.jpg"),
        width=1920,
        height=1500,
        mime="image/jpeg",
        license_short="Public domain",
        creator="Sam Shere",
        page_url="https://commons.wikimedia.org/wiki/File:Hindenburg.jpg",
    )


def test_normalise_hashtag_lowercases_and_prefixes() -> None:
    assert _normalise_hashtag("History") == "#history"
    assert _normalise_hashtag("#TrueCrime") == "#truecrime"
    assert _normalise_hashtag("  weird stuff  ") == "#weirdstuff"
    assert _normalise_hashtag("") == ""


def test_attribution_block_dedupes_assets() -> None:
    a = _sample_asset()
    out = _attribution_block([a, a, a], _sample_script())
    assert out.count("Sam Shere") == 1
    assert "wikipedia.org/wiki/Hindenburg_disaster" in out


def test_render_metadata_text_includes_all_fields() -> None:
    meta = ShortMetadata(
        title="What sank the Hindenburg in seconds?",
        description="In 1937 a luxury airship became a fireball.",
        hashtags=["#truecrime", "#history"],
        full_description="In 1937 a luxury airship became a fireball.\n\n#truecrime #history",
    )
    text = render_metadata_text(meta)
    assert "TITLE" in text
    assert meta.title in text
    assert "#truecrime #history" in text


def test_generate_metadata_calls_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    payload = {
        "title": "What sank the Hindenburg in 30 seconds?",
        "description": "In 1937 a German airship turned into a fireball above New Jersey. Watch how it ended an era.",
        "hashtags": ["truecrime", "#disasters", "#hindenburg"],
    }
    fake_resp = MagicMock()
    fake_resp.text = json.dumps(payload)
    fake_models = MagicMock()
    fake_models.generate_content.return_value = fake_resp
    fake_client = MagicMock()
    fake_client.models = fake_models
    with patch("google.genai.Client", return_value=fake_client):
        meta = generate_metadata(_sample_script(), [_sample_asset()])

    assert meta.title.startswith("What sank")
    assert "#truecrime" in meta.hashtags
    assert "#disasters" in meta.hashtags
    assert "Sam Shere" in meta.full_description
    assert "Hindenburg_disaster" in meta.full_description
