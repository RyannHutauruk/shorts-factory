"""Tests for the local scheduler: config parsing, defaults, queue refill."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from shorts_factory.upload.scheduler import (
    ALL_NICHES,
    ScheduleConfig,
    _queue_is_empty,
    load_config,
    systemd_unit,
    write_default_config,
)


def test_default_config_written(tmp_path: Path) -> None:
    p = tmp_path / "schedule.toml"
    write_default_config(p)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "slots" in content
    assert "niche" in content
    assert "upload" in content


def test_default_config_not_overwritten(tmp_path: Path) -> None:
    p = tmp_path / "schedule.toml"
    p.write_text("custom = true\n", encoding="utf-8")
    write_default_config(p)
    assert p.read_text(encoding="utf-8") == "custom = true\n"


def test_default_config_force_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "schedule.toml"
    p.write_text("custom = true\n", encoding="utf-8")
    write_default_config(p, force=True)
    assert "slots" in p.read_text(encoding="utf-8")


def test_load_config_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    p = tmp_path / "schedule.toml"
    p.write_text(
        'slots = ["08:00"]\n'
        'queue_path = "~/queue.txt"\n'
        'out_root = "~/out"\n'
        'history_path = "~/.config/sf/history.json"\n'
        'niche = "science"\n'
        "upload = false\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.slots == ["08:00"]
    assert str(cfg.queue_path).startswith(str(fake_home))
    assert cfg.upload is False
    assert cfg.niche == "science"


def test_load_config_falls_back_to_defaults(tmp_path: Path) -> None:
    p = tmp_path / "schedule.toml"
    p.write_text("", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.slots == ["09:30", "20:00"]
    assert cfg.niche == "history"
    assert cfg.upload is True


def test_queue_is_empty_treats_comments_as_empty(tmp_path: Path) -> None:
    p = tmp_path / "queue.txt"
    p.write_text("# done @ ...: foo\n# foo\n  \n", encoding="utf-8")
    assert _queue_is_empty(p) is True


def test_queue_is_empty_false_with_real_topic(tmp_path: Path) -> None:
    p = tmp_path / "queue.txt"
    p.write_text("# old\nHalifax explosion\n", encoding="utf-8")
    assert _queue_is_empty(p) is False


def test_systemd_unit_includes_exec_and_config(tmp_path: Path) -> None:
    cfg = tmp_path / "schedule.toml"
    cfg.write_text("slots=[]\n", encoding="utf-8")
    out = systemd_unit(cfg)
    assert "shorts-factory local scheduler" in out
    assert "schedule run" in out
    assert "GEMINI_API_KEY=" in out


# ----- multi-niche rotation + audience -------------------------------------


def test_load_config_accepts_niche_list(tmp_path: Path) -> None:
    p = tmp_path / "schedule.toml"
    p.write_text(
        'niche = ["history", "science", "space"]\naudience = "US"\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.niche == ["history", "science", "space"]
    assert cfg.audience == "US"


def test_niche_list_resolves_all() -> None:
    cfg = ScheduleConfig(niche="all")
    assert cfg.niche_list() == list(ALL_NICHES)


def test_niche_list_resolves_single() -> None:
    cfg = ScheduleConfig(niche="history")
    assert cfg.niche_list() == ["history"]


def test_niche_list_filters_unknown() -> None:
    cfg = ScheduleConfig(niche=["history", "not_a_niche", "science"])
    assert cfg.niche_list() == ["history", "science"]


def test_niche_list_rejects_unknown_string() -> None:
    cfg = ScheduleConfig(niche="not_a_niche")
    with pytest.raises(ValueError):
        cfg.niche_list()


def test_pick_niche_deterministic_for_single() -> None:
    cfg = ScheduleConfig(niche="history")
    rng = random.Random(0)
    assert cfg.pick_niche(rng=rng) == "history"


def test_pick_niche_avoids_back_to_back() -> None:
    """With 2+ niches, the most recent niche should not be picked again."""
    cfg = ScheduleConfig(niche=["history", "science"])
    rng = random.Random(42)
    cfg.last_niche = "history"
    # All draws from a 1-element candidate pool must equal "science"
    for _ in range(20):
        assert cfg.pick_niche(rng=rng) == "science"


def test_pick_niche_uniform_across_rotation() -> None:
    """Without recent-niche bias, each niche in the rotation is reachable."""
    cfg = ScheduleConfig(niche=["history", "science", "space"])
    rng = random.Random(0)
    seen = {cfg.pick_niche(rng=rng) for _ in range(100)}
    assert seen == {"history", "science", "space"}


def test_queue_path_for_single_niche_returns_base(tmp_path: Path) -> None:
    cfg = ScheduleConfig(niche="history", queue_path=tmp_path / "queue.txt")
    assert cfg.queue_path_for("history") == tmp_path / "queue.txt"


def test_queue_path_for_multi_niche_uses_per_niche_file(tmp_path: Path) -> None:
    cfg = ScheduleConfig(niche=["history", "science"], queue_path=tmp_path / "queue.txt")
    assert cfg.queue_path_for("history") == tmp_path / "queue_history.txt"
    assert cfg.queue_path_for("science") == tmp_path / "queue_science.txt"


def test_queue_path_for_all_uses_per_niche_file(tmp_path: Path) -> None:
    cfg = ScheduleConfig(niche="all", queue_path=tmp_path / "queue.txt")
    assert cfg.queue_path_for("space") == tmp_path / "queue_space.txt"
