"""Tests for niche.longform_metadata (Gemini call is mocked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from shorts_factory.niche.broll import BrollAsset
from shorts_factory.niche.longform_metadata import (
    LongformMetadata,
    _attribution_block,
    _normalise_hashtag,
    _validate_meta_payload,
    generate_longform_metadata,
    render_longform_metadata_text,
)
from shorts_factory.niche.longform_script import (
    LongformBeat,
    LongformChapter,
    LongformScript,
)
from shorts_factory.niche.longform_voice import (
    LongformChapterAudio,
    LongformNarration,
    LongformParagraphSegment,
)


def _asset(name: str = "img1", *, page_url: str | None = None) -> BrollAsset:
    return BrollAsset(
        query=name,
        title=name,
        local_path=Path(f"/tmp/{name}.jpg"),
        width=1920,
        height=1080,
        mime="image/jpeg",
        license_short="CC BY-SA 4.0",
        creator="Anon",
        page_url=page_url or f"https://commons.wikimedia.org/wiki/File:{name}",
    )


def _make_script() -> LongformScript:
    return LongformScript(
        topic="Halifax explosion",
        niche="history",
        cold_open="A massive blast tore through the harbour.",
        chapters=[
            LongformChapter(
                index=1,
                title="The collision",
                summary="Two ships meet in the Narrows.",
                transition="t",
                beats=[LongformBeat(text="b1", visual_hint="ships")],
                closing="c",
            ),
            LongformChapter(
                index=2,
                title="The blast",
                summary="The largest pre-nuclear explosion.",
                transition="t",
                beats=[LongformBeat(text="b2", visual_hint="explosion")],
                closing="c",
            ),
        ],
        outro="A city rebuilt itself.",
        sources=["Wikipedia: Halifax Explosion"],
    )


def _make_narration(tmp_path: Path) -> LongformNarration:
    audio = tmp_path / "full.wav"
    audio.write_bytes(b"\x00" * 100)
    cold = LongformParagraphSegment(text="cold", start=0.0, end=20.0, audio_path=audio)
    ch1_para = LongformParagraphSegment(text="b1", start=20.0, end=80.0, audio_path=audio)
    ch2_para = LongformParagraphSegment(text="b2", start=80.0, end=160.0, audio_path=audio)
    outro = LongformParagraphSegment(text="o", start=160.0, end=180.0, audio_path=audio)
    chapters = [
        LongformChapterAudio(
            chapter_index=1,
            title="The collision",
            voice_path=audio,
            paragraphs=[ch1_para],
            audio_path=audio,
            duration=60.0,
            start=20.0,
            end=80.0,
        ),
        LongformChapterAudio(
            chapter_index=2,
            title="The blast",
            voice_path=audio,
            paragraphs=[ch2_para],
            audio_path=audio,
            duration=80.0,
            start=80.0,
            end=160.0,
        ),
    ]
    return LongformNarration(
        full_audio=audio,
        duration=180.0,
        cold_open=cold,
        chapters=chapters,
        outro=outro,
    )


def test_validate_meta_payload_ok() -> None:
    _validate_meta_payload(
        {"title": "x", "description": "y", "hashtags": ["#a"]},
    )


def test_validate_meta_payload_missing_key() -> None:
    import pytest

    with pytest.raises(ValueError, match="title"):
        _validate_meta_payload({"description": "x", "hashtags": []})


def test_normalise_hashtag_adds_hash_and_lowercases() -> None:
    assert _normalise_hashtag("Documentary") == "#documentary"
    assert _normalise_hashtag("#History") == "#history"
    assert _normalise_hashtag("multi word") == "#multiword"
    assert _normalise_hashtag("") == ""


def test_attribution_block_dedupes_by_page_url() -> None:
    a = _asset("a")
    b = _asset("b")
    duplicate = _asset("a-copy", page_url=a.page_url)
    script = _make_script()
    block = _attribution_block([a, b, duplicate], script)
    # Only 2 unique credit lines (a and b), even though we passed 3 assets.
    assert block.count("•") == 3  # 2 image lines + 1 source line
    assert "Halifax Explosion" in block


def test_render_longform_metadata_text_layout() -> None:
    meta = LongformMetadata(
        title="t",
        description="d",
        hashtags=["#one", "#two"],
        full_description="d\n\n00:00 Intro\n\n#one #two",
    )
    text = render_longform_metadata_text(meta)
    assert text.startswith("TITLE\nt")
    assert "DESCRIPTION\nd\n\n00:00 Intro" in text
    assert "HASHTAGS\n#one #two" in text


def test_generate_longform_metadata_mocked(tmp_path: Path) -> None:
    """Mock Gemini response and check the assembled metadata."""
    script = _make_script()
    narration = _make_narration(tmp_path)
    visuals = [_asset("a"), _asset("b")]

    fake_response = MagicMock()
    fake_response.text = json.dumps(
        {
            "title": "The Halifax explosion: a city's worst day",
            "description": "On a December morning a collision in the harbour "
            "set off the largest pre-nuclear blast in history. "
            "What followed reshaped Halifax forever. Tell us "
            "what surprised you in the comments.",
            "hashtags": ["#documentary", "#history", "#halifax"],
        }
    )

    fake_client = MagicMock()
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(genai=fake_genai),
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        },
    ):
        with patch(
            "shorts_factory.niche.script._gemini_call_with_retry",
            return_value=fake_response,
        ):
            meta = generate_longform_metadata(
                script,
                narration,
                visuals,
                api_key="fake_key",
            )

    assert meta.title.startswith("The Halifax")
    assert "#documentary" in meta.hashtags
    assert "00:00" in meta.full_description  # cold open chapter timestamp
    assert "Image credits" in meta.full_description
    assert "Sources" in meta.full_description
