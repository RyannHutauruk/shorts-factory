"""Tests for niche.batch: history persistence, queue, anti-repetition."""

from __future__ import annotations

from pathlib import Path

from shorts_factory.niche.batch import BatchHistory, parse_topics_file, pop_next_topic


def test_history_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "history.json"
    h = BatchHistory.load(p)
    h.record("Hindenburg disaster", "What caused the unsinkable airship to erupt?")
    h.save()
    h2 = BatchHistory.load(p)
    assert h2.has_topic("Hindenburg disaster")
    assert h2.has_topic("hindenburg DISASTER")  # case-insensitive
    assert h2.has_hook_stem("What caused the unsinkable airship to do anything?")


def test_history_hook_stem_is_first_six_words(tmp_path: Path) -> None:
    h = BatchHistory(path=tmp_path / "h.json")
    # Alpha-only, lowercased, first 6 tokens.
    assert h.hook_stem("Two jets collided in fog") == "two jets collided in fog"
    assert (
        h.hook_stem("Two jumbo jets collided on a foggy runway.") == "two jumbo jets collided on a"
    )
    # Digits are ignored - they are not part of the alpha stem.
    assert (
        h.hook_stem("In 1986 a reactor exploded silently overnight")
        == "in a reactor exploded silently overnight"
    )


def test_history_caps_recent_stems(tmp_path: Path) -> None:
    h = BatchHistory(path=tmp_path / "h.json", max_recent_stems=3)
    # Distinct alpha words per hook so stems don't collide (digits are ignored).
    distinct = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
    ]
    for i, w in enumerate(distinct):
        h.record(f"topic-{i}", f"hook {w} sentence keeps going forward fast")
    h.save()
    h2 = BatchHistory.load(tmp_path / "h.json")
    assert len(h2.hook_stems) == 3
    # Most recent ones survive
    assert "hook juliet sentence keeps going forward" in h2.hook_stems


def test_parse_topics_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / "queue.txt"
    f.write_text(
        "# this is a comment\n"
        "\n"
        "Hindenburg disaster\n"
        "  # done @ 2025-01-01: old item\n"
        "Tenerife airport disaster\n",
        encoding="utf-8",
    )
    topics = parse_topics_file(f)
    assert topics == ["Hindenburg disaster", "Tenerife airport disaster"]


def test_pop_next_topic_marks_taken_in_place(tmp_path: Path) -> None:
    f = tmp_path / "queue.txt"
    f.write_text("Hindenburg disaster\nTenerife airport disaster\n", encoding="utf-8")
    h = BatchHistory(path=tmp_path / "history.json")
    first = pop_next_topic(f, h)
    assert first == "Hindenburg disaster"
    contents = f.read_text(encoding="utf-8")
    assert "# done @" in contents
    assert "Hindenburg disaster" in contents  # commented, but still present
    assert "Tenerife airport disaster" in contents
    # next pop returns the second (first is now commented)
    second = pop_next_topic(f, h)
    assert second == "Tenerife airport disaster"


def test_pop_next_topic_skips_already_completed(tmp_path: Path) -> None:
    f = tmp_path / "queue.txt"
    f.write_text("Hindenburg disaster\nTenerife airport disaster\n", encoding="utf-8")
    h = BatchHistory(path=tmp_path / "history.json")
    h.record("Hindenburg disaster", "hook one")
    out = pop_next_topic(f, h)
    assert out == "Tenerife airport disaster"


def test_pop_next_topic_returns_none_when_queue_empty(tmp_path: Path) -> None:
    f = tmp_path / "queue.txt"
    f.write_text("# done @ ...: foo\n", encoding="utf-8")
    h = BatchHistory(path=tmp_path / "history.json")
    assert pop_next_topic(f, h) is None
