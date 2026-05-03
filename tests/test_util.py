"""Unit tests for util helpers."""

from __future__ import annotations

from shorts_factory.util import format_timestamp, slugify


def test_slugify_handles_punctuation() -> None:
    assert slugify("Night of the Living Dead!") == "night-of-the-living-dead"
    assert slugify("  ") == "untitled"


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "0:00:00.000"
    assert format_timestamp(75.25) == "0:01:15.250"
    assert format_timestamp(3661.5) == "1:01:01.500"
    assert format_timestamp(-5) == "0:00:00.000"
