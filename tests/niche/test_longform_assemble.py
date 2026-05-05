"""Tests for niche.longform_assemble (no ffmpeg invocation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shorts_factory.niche.broll import BrollAsset
from shorts_factory.niche.longform_assemble import (
    CHAPTER_CARD_DURATION,
    LongformShot,
    _allocate_shots,
    _shot_filter,
    _split_paragraph_into_phrases,
    build_caption_ass,
    build_shot_list,
    chapter_timestamps_block,
)
from shorts_factory.niche.longform_broll import (
    LongformBrollPlan,
    LongformChapterBroll,
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


def _asset(name: str) -> BrollAsset:
    return BrollAsset(
        query=name,
        title=name,
        local_path=Path(f"/tmp/{name}.jpg"),
        width=1920,
        height=1080,
        mime="image/jpeg",
        license_short="CC BY-SA 4.0",
        creator="Anon",
        page_url=f"https://commons.wikimedia.org/wiki/File:{name}",
    )


def _seg(text: str, start: float, end: float, *, name: str = "x") -> LongformParagraphSegment:
    return LongformParagraphSegment(
        text=text,
        audio_path=Path(f"/tmp/{name}.wav"),
        start=start,
        end=end,
    )


# ---------- _split_paragraph_into_phrases ----------


def test_split_paragraph_returns_empty_for_empty_text() -> None:
    assert _split_paragraph_into_phrases("", start=0, end=5) == []


def test_split_paragraph_phrases_span_the_full_duration() -> None:
    text = "this is a sentence with several words used to test caption phrase splitting"
    phrases = _split_paragraph_into_phrases(text, start=10.0, end=20.0)
    assert phrases
    # First phrase starts at 10.0, last phrase ends at 20.0
    assert phrases[0][0] == pytest.approx(10.0, abs=1e-3)
    assert phrases[-1][1] == pytest.approx(20.0, abs=1e-3)
    # Each phrase is non-empty
    for s, e, t in phrases:
        assert e > s
        assert t.strip()


# ---------- _allocate_shots ----------


def test_allocate_shots_caps_each_shot_to_max_duration() -> None:
    """A 30s paragraph should produce ~5-6 shots (one per ~5.5s) to avoid
    holding a single image for half a minute (the previous behaviour)."""
    seg = _seg("hello world", start=0.0, end=30.0)
    assets = [_asset("a"), _asset("b"), _asset("c")]
    seen: set[str] = set()
    shots = _allocate_shots(seg, assets, [], seen_in_section=seen)
    assert 4 <= len(shots) <= 7, f"expected 4-7 shots, got {len(shots)}"
    # Last shot's end should equal the segment's end
    assert shots[-1].end == pytest.approx(seg.end, abs=1e-3)
    # All shots together cover the segment
    total = sum(s.duration for s in shots)
    assert total == pytest.approx(seg.duration, abs=1e-3)
    # Each shot honours the per-shot duration cap (within rounding).
    for s in shots:
        assert s.duration <= 7.0, f"shot too long: {s.duration}"


def test_allocate_shots_short_segment_emits_single_shot() -> None:
    """A short paragraph (under MAX_SHOT_DURATION) gets one shot covering it."""
    seg = _seg("hi", start=0.0, end=4.0)
    shots = _allocate_shots(seg, [_asset("a")], [], seen_in_section=set())
    assert len(shots) == 1
    assert shots[0].duration == pytest.approx(4.0, abs=1e-3)


def test_allocate_shots_cycles_through_pool_when_more_shots_than_assets() -> None:
    """If a 30s paragraph only has 2 assets, shots cycle a,b,a,b,a,b."""
    seg = _seg("long", start=0.0, end=30.0)
    assets = [_asset("a"), _asset("b")]
    shots = _allocate_shots(seg, assets, [], seen_in_section=set())
    paths = [s.image_path.stem for s in shots]
    assert paths[0] == "a"
    assert paths[1] == "b"
    # Cycling means we see each asset multiple times.
    assert paths.count("a") >= 2
    assert paths.count("b") >= 2


def test_allocate_shots_rotates_motion_presets() -> None:
    """Consecutive shots within a segment use different motion presets."""
    seg = _seg("long", start=0.0, end=30.0)
    assets = [_asset(f"img{i}") for i in range(8)]
    shots = _allocate_shots(seg, assets, [], seen_in_section=set())
    motions = [s.motion for s in shots]
    # First two shots should differ.
    assert motions[0] != motions[1]
    # Across all shots we should see at least 3 distinct motions.
    assert len(set(motions)) >= 3


def test_allocate_shots_falls_back_when_assets_empty() -> None:
    seg = _seg("hello", start=0.0, end=3.0)
    fallbacks = [_asset("fallback")]
    seen: set[str] = set()
    shots = _allocate_shots(seg, [], fallbacks, seen_in_section=seen)
    assert len(shots) >= 1
    assert shots[0].image_path == fallbacks[0].local_path


def test_allocate_shots_raises_when_no_assets_and_no_fallbacks() -> None:
    seg = _seg("hi", start=0.0, end=2.0)
    seen: set[str] = set()
    with pytest.raises(RuntimeError, match="no B-roll asset"):
        _allocate_shots(seg, [], [], seen_in_section=seen)


# ---------- build_shot_list ----------


def _make_chapter_audio(
    idx: int, *, start: float, beats: int = 2
) -> tuple[LongformChapterAudio, list[LongformParagraphSegment]]:
    cursor = start
    segs: list[LongformParagraphSegment] = []
    # transition
    transition = LongformParagraphSegment(
        text=f"chapter {idx} transition",
        audio_path=Path(f"/tmp/chapter_{idx:02d}/transition.wav"),
        start=cursor,
        end=cursor + 4.0,
    )
    segs.append(transition)
    cursor += 4.0
    # beats
    for j in range(beats):
        beat_seg = LongformParagraphSegment(
            text=f"beat {j} of chapter {idx}",
            audio_path=Path(f"/tmp/chapter_{idx:02d}/beat_{j:02d}.wav"),
            start=cursor,
            end=cursor + 8.0,
        )
        segs.append(beat_seg)
        cursor += 8.0
    closing = LongformParagraphSegment(
        text=f"chapter {idx} closing",
        audio_path=Path(f"/tmp/chapter_{idx:02d}/closing.wav"),
        start=cursor,
        end=cursor + 3.0,
    )
    segs.append(closing)
    cursor += 3.0
    chap_audio = LongformChapterAudio(
        chapter_index=idx,
        title=f"Chapter {idx}",
        audio_path=Path(f"/tmp/chapter_{idx:02d}/chapter.wav"),
        duration=cursor - start,
        start=start,
        end=cursor,
        paragraphs=segs,
        voice_path=Path("/tmp/voice.onnx"),
    )
    return chap_audio, segs


def _make_full_run() -> tuple[LongformScript, LongformNarration, LongformBrollPlan]:
    cold_open_seg = _seg("cold open hook", 0.0, 5.0, name="cold_open")
    chap1_audio, _ = _make_chapter_audio(1, start=5.0, beats=2)
    chap2_audio, _ = _make_chapter_audio(2, start=chap1_audio.end + 0.5, beats=2)
    outro_seg = _seg("outro", chap2_audio.end + 0.5, chap2_audio.end + 5.5, name="outro")

    narration = LongformNarration(
        full_audio=Path("/tmp/narration.wav"),
        duration=outro_seg.end,
        cold_open=cold_open_seg,
        chapters=[chap1_audio, chap2_audio],
        outro=outro_seg,
    )

    chapters_script = [
        LongformChapter(
            index=i,
            title=f"Chapter {i}",
            summary="s",
            transition=f"chapter {i} transition",
            beats=[
                LongformBeat(text=f"beat {j} of chapter {i}", visual_hint="x") for j in range(2)
            ],
            closing=f"chapter {i} closing",
        )
        for i in (1, 2)
    ]
    script = LongformScript(
        topic="Halifax explosion",
        niche="history",
        cold_open="cold open hook",
        chapters=chapters_script,
        outro="outro",
    )

    broll = LongformBrollPlan(
        cold_open_assets=[_asset(f"cold_{i}") for i in range(3)],
        chapters=[
            LongformChapterBroll(
                chapter_index=i,
                title=f"Chapter {i}",
                beat_assets=[
                    [_asset(f"ch{i}_beat0_a"), _asset(f"ch{i}_beat0_b")],
                    [_asset(f"ch{i}_beat1_a")],
                ],
                chapter_assets=[_asset(f"ch{i}_fallback")],
            )
            for i in (1, 2)
        ],
        outro_assets=[_asset("outro_a")],
    )
    return script, narration, broll


def test_build_shot_list_covers_entire_timeline() -> None:
    script, narration, broll = _make_full_run()
    shots = build_shot_list(script, narration, broll)
    # First shot starts at the cold open's start (0.0)
    assert shots[0].start == pytest.approx(0.0, abs=1e-3)
    # Last shot ends at or near the outro's end
    assert shots[-1].end == pytest.approx(narration.outro.end, abs=0.1)
    # Shots are monotonic
    for prev, curr in zip(shots, shots[1:], strict=False):
        assert curr.start >= prev.end - 0.05


def test_build_shot_list_uses_fallback_when_beat_assets_empty() -> None:
    """A chapter beat with no assets should still produce a shot via fallback."""
    script, narration, broll = _make_full_run()

    # Wipe one beat's assets in the first chapter; this beat must still get a shot.
    plan = LongformBrollPlan(
        cold_open_assets=broll.cold_open_assets,
        chapters=[
            LongformChapterBroll(
                chapter_index=broll.chapters[0].chapter_index,
                title=broll.chapters[0].title,
                beat_assets=[[], broll.chapters[0].beat_assets[1]],  # beat 0 EMPTY
                chapter_assets=broll.chapters[0].chapter_assets,
            ),
            broll.chapters[1],
        ],
        outro_assets=broll.outro_assets,
    )
    shots = build_shot_list(script, narration, plan)
    # No silent gaps in the timeline
    for prev, curr in zip(shots, shots[1:], strict=False):
        assert curr.start >= prev.end - 0.05


# ---------- ASS captions ----------


def test_build_caption_ass_includes_chapter_cards() -> None:
    script, narration, broll = _make_full_run()
    ass = build_caption_ass(script, narration, title="Halifax", show_chapter_cards=True)
    # One ChapterCard line per chapter
    assert ass.count("ChapterCard") == len(narration.chapters) + 1  # +1 for the style def
    assert "Chapter 1 - Chapter 1" in ass
    assert "Halifax" in ass


def test_build_caption_ass_skips_chapter_cards_when_disabled() -> None:
    script, narration, broll = _make_full_run()
    ass = build_caption_ass(script, narration, show_chapter_cards=False)
    # The Style line is still in the header, but no ChapterCard Dialogue lines.
    dialogue_chapter = [
        ln for ln in ass.splitlines() if ln.startswith("Dialogue") and "ChapterCard" in ln
    ]
    assert len(dialogue_chapter) == 0


def test_build_caption_ass_has_paragraph_caption_lines() -> None:
    script, narration, broll = _make_full_run()
    ass = build_caption_ass(script, narration)
    caption_lines = [
        ln for ln in ass.splitlines() if ln.startswith("Dialogue") and ",Caption," in ln
    ]
    # At least one caption per paragraph (cold open + 8 chapter paragraphs + outro)
    assert len(caption_lines) >= 10


def test_chapter_card_duration_constant_is_short() -> None:
    assert 1.5 <= CHAPTER_CARD_DURATION <= 5.0


# ---------- chapter timestamps ----------


def test_chapter_timestamps_block_starts_with_zero() -> None:
    script, narration, broll = _make_full_run()
    block = chapter_timestamps_block(narration)
    lines = block.splitlines()
    assert lines[0] == "00:00 Cold open"
    # Subsequent lines mention chapter title and a timestamp ahead of 0
    assert any("Chapter 1" in ln for ln in lines)
    assert any("Chapter 2" in ln for ln in lines)


# ---------- shot dataclass ----------


def test_longform_shot_duration_matches_end_minus_start() -> None:
    shot = LongformShot(image_path=Path("/x"), duration=2.5, start=10.0, end=12.5)
    assert shot.end - shot.start == pytest.approx(shot.duration, abs=1e-9)


# ---------- _shot_filter (ffmpeg filter graph) ----------


def test_shot_filter_uses_fit_inside_canvas_strategy() -> None:
    """The foreground must use ``force_original_aspect_ratio=decrease`` so an
    ultra-wide source image never overflows the 1920x1080 canvas (regression
    test for "Padded dimensions cannot be smaller than input dimensions")."""
    g = _shot_filter(0, 2.0, "shot0")
    assert "force_original_aspect_ratio=decrease" in g
    # And the background still uses 'increase' for the blurred-bars effect.
    assert "force_original_aspect_ratio=increase" in g
    # Padding always goes to 1920x1080, never smaller.
    assert "pad=1920:1080" in g


@pytest.mark.parametrize(
    "motion",
    ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"],
)
def test_shot_filter_supports_all_motion_presets(motion: str) -> None:
    g = _shot_filter(0, 4.0, "shot0", motion=motion)
    # Every motion still emits a zoompan filter on the foreground stack.
    assert "zoompan=z=" in g
