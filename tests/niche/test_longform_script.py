"""Tests for niche.longform_script: outline + chapter expansion + script roundtrip."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shorts_factory.niche.longform_script import (
    LongformBeat,
    LongformChapter,
    LongformScript,
    _resolve_chapter_count,
    _resolve_total_words,
    _validate_chapter_payload,
    _validate_outline_payload,
    expand_chapter,
    generate_longform_script,
    generate_outline,
    script_from_dict,
    script_from_file,
    script_to_dict,
)

# ---------- Pure-Python helpers ----------


def test_chapter_count_scales_with_duration() -> None:
    assert _resolve_chapter_count(5) == 4  # clamped to floor
    assert _resolve_chapter_count(10) == 5
    assert _resolve_chapter_count(14) == 7
    assert _resolve_chapter_count(25) == 7  # clamped to ceiling


def test_total_words_scales_with_duration() -> None:
    assert _resolve_total_words(10) == 1560  # 10 * 60 * 2.6
    assert _resolve_total_words(5) == 780


# ---------- Validation ----------


def test_validate_outline_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        _validate_outline_payload([], 5)  # type: ignore[arg-type]


def test_validate_outline_rejects_missing_keys() -> None:
    with pytest.raises(ValueError, match="cold_open_summary"):
        _validate_outline_payload({"chapters": [{}], "outro_summary": ""}, 5)


def test_validate_outline_rejects_wrong_chapter_count() -> None:
    payload = {
        "cold_open_summary": "x",
        "outro_summary": "x",
        "chapters": [{"title": "t", "summary": "s"}],  # 1 chapter, expected ~5
    }
    with pytest.raises(ValueError, match="chapters; expected"):
        _validate_outline_payload(payload, 5)


def test_validate_outline_accepts_close_chapter_count() -> None:
    chapters = [{"title": f"c{i}", "summary": "s"} for i in range(4)]
    payload = {
        "cold_open_summary": "x",
        "outro_summary": "x",
        "chapters": chapters,
    }
    # 4 chapters when target is 5 should be allowed (within ±2 tolerance).
    _validate_outline_payload(payload, 5)


def test_validate_chapter_rejects_empty_beats() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _validate_chapter_payload({"transition": "", "beats": [], "closing": ""})


def test_validate_chapter_requires_text_and_visual_hint() -> None:
    payload: dict[str, Any] = {
        "transition": "t",
        "beats": [{"text": "only text"}],
        "closing": "c",
    }
    with pytest.raises(ValueError, match="visual_hint"):
        _validate_chapter_payload(payload)


# ---------- LongformScript dataclass ----------


def _make_chapter(idx: int = 1, *, beats: int = 3) -> LongformChapter:
    return LongformChapter(
        index=idx,
        title=f"Chapter {idx}",
        summary="summary",
        transition="trans " * 5,  # 5 words
        beats=[
            LongformBeat(text="beat text " * 30, visual_hint=f"visual {i}") for i in range(beats)
        ],
        closing="closing line " * 5,
    )


def test_chapter_word_count_and_estimated_seconds() -> None:
    ch = _make_chapter()
    # transition (10) + 3 beats * 60 + closing (10) = 200 words approx
    assert ch.word_count > 100
    assert ch.estimated_seconds > 30


def test_longform_script_chapter_timestamps_are_cumulative() -> None:
    chapters = [_make_chapter(i, beats=3) for i in range(1, 4)]
    script = LongformScript(
        topic="t",
        niche="history",
        cold_open="cold open " * 40,  # 80 words ~ 30s
        chapters=chapters,
        outro="outro " * 40,
    )
    timestamps = script.chapter_timestamps
    # First entry is cold open at 0:00
    assert timestamps[0] == (0, "Cold open")
    # Subsequent timestamps strictly increase
    for prev, curr in zip(timestamps, timestamps[1:], strict=False):
        assert curr[0] > prev[0]
    # Number of entries = 1 (cold open) + N chapters
    assert len(timestamps) == 1 + len(chapters)


def test_longform_script_all_text_concatenates_in_order() -> None:
    chapters = [_make_chapter(i, beats=2) for i in range(1, 3)]
    script = LongformScript(
        topic="t",
        niche="science",
        cold_open="THE_OPEN",
        chapters=chapters,
        outro="THE_END",
    )
    text = script.all_text
    assert text.startswith("THE_OPEN")
    assert text.endswith("THE_END")
    # Chapter 1 transition appears before chapter 2 transition
    idx_ch1 = text.find("trans")  # first occurrence
    idx_close1 = text.find("closing line")
    assert idx_ch1 < idx_close1


# ---------- Roundtrip ----------


def test_script_to_dict_and_back_roundtrips() -> None:
    chapters = [_make_chapter(1, beats=2), _make_chapter(2, beats=3)]
    script = LongformScript(
        topic="Halifax explosion",
        niche="history",
        cold_open="cold",
        chapters=chapters,
        outro="outro",
        sources=["wikipedia.org/wiki/Halifax_explosion"],
        duration_target_min=12.0,
    )
    rebuilt = script_from_dict(script_to_dict(script))
    assert rebuilt.topic == script.topic
    assert rebuilt.duration_target_min == 12.0
    assert len(rebuilt.chapters) == 2
    assert rebuilt.chapters[1].beats[0].visual_hint == script.chapters[1].beats[0].visual_hint
    assert rebuilt.sources == script.sources


def test_script_from_file(tmp_path: Path) -> None:
    chapters = [_make_chapter(1, beats=2)]
    script = LongformScript(
        topic="test",
        niche="history",
        cold_open="open",
        chapters=chapters,
        outro="end",
    )
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script_to_dict(script)), encoding="utf-8")

    loaded = script_from_file(path)
    assert loaded.topic == "test"
    assert loaded.chapters[0].beats[0].text == script.chapters[0].beats[0].text


# ---------- Gemini integration (mocked) ----------


def _fake_resp(payload: dict[str, Any]) -> Any:
    resp = MagicMock()
    resp.text = json.dumps(payload)
    return resp


def _outline_payload(n_chapters: int = 5) -> dict[str, Any]:
    return {
        "cold_open_summary": "What if a single radio call could have saved hundreds?",
        "chapters": [
            {
                "title": f"Chapter {i + 1}",
                "summary": f"Covers part {i + 1} of the story",
                "key_points": ["fact a", "fact b", "fact c"],
                "broll_keywords": ["foo", "bar", "baz"],
            }
            for i in range(n_chapters)
        ],
        "outro_summary": "And that's why the lesson endures.",
        "sources": ["wikipedia.org/wiki/X"],
    }


def _chapter_payload() -> dict[str, Any]:
    return {
        "transition": "It began on a clear morning in November.",
        "beats": [
            {
                "text": "Halifax harbor was busy with wartime shipping.",
                "visual_hint": "Halifax harbor 1917",
            },
            {
                "text": "The SS Mont-Blanc carried 2,925 tonnes of explosives.",
                "visual_hint": "SS Mont-Blanc",
            },
            {
                "text": "It collided with the Norwegian SS Imo at 8:45 AM.",
                "visual_hint": "Halifax explosion debris field",
            },
        ],
        "closing": "What followed would redraw the city forever.",
    }


def test_generate_outline_calls_gemini_and_returns_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_resp(_outline_payload(5))

    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(genai=fake_genai),
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        },
    ):
        outline = generate_outline("Halifax explosion", niche="history", duration_min=10)

    assert "chapters" in outline
    assert len(outline["chapters"]) == 5
    sent_prompt = fake_client.models.generate_content.call_args.kwargs["contents"]
    assert "Halifax explosion" in sent_prompt
    assert "history" in sent_prompt
    assert "United States" in sent_prompt or "US viewers" in sent_prompt


def test_generate_outline_rejects_unknown_niche() -> None:
    with pytest.raises(ValueError, match="unknown niche"):
        generate_outline("topic", niche="cooking_recipes")


def test_generate_outline_rejects_unknown_audience() -> None:
    with pytest.raises(ValueError, match="unknown audience"):
        generate_outline("topic", niche="history", audience="MARS")


def test_expand_chapter_returns_longform_chapter(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = _fake_resp(_chapter_payload())

    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(genai=fake_genai),
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        },
    ):
        ch = expand_chapter(
            topic="Halifax explosion",
            niche="history",
            chapter_index=2,
            chapter_title="The Collision",
            chapter_summary="Two ships collide in the harbor",
            key_points=["fact 1", "fact 2"],
            duration_min=10,
        )

    assert ch.index == 2
    assert ch.title == "The Collision"
    assert len(ch.beats) == 3
    assert ch.beats[1].visual_hint == "SS Mont-Blanc"


def test_generate_longform_script_combines_outline_and_chapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: outline call + N chapter expansion calls -> LongformScript."""
    fake_client = MagicMock()
    # First call returns the outline; subsequent calls return per-chapter expansion.
    fake_client.models.generate_content.side_effect = [
        _fake_resp(_outline_payload(4)),
        _fake_resp(_chapter_payload()),
        _fake_resp(_chapter_payload()),
        _fake_resp(_chapter_payload()),
        _fake_resp(_chapter_payload()),
    ]

    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    fake_types.GenerateContentConfig = MagicMock()

    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    with patch.dict(
        "sys.modules",
        {
            "google": MagicMock(genai=fake_genai),
            "google.genai": fake_genai,
            "google.genai.types": fake_types,
        },
    ):
        script = generate_longform_script("Halifax explosion", niche="history", duration_min=8)

    # Outline (1) + chapter expansions (4) = 5 calls
    assert fake_client.models.generate_content.call_count == 5
    assert script.topic == "Halifax explosion"
    assert len(script.chapters) == 4
    assert script.cold_open.startswith("What if")
    assert script.chapters[0].beats[1].visual_hint == "SS Mont-Blanc"
