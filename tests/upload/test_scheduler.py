"""Tests for the local scheduler: config parsing, defaults, queue refill."""

from __future__ import annotations

from pathlib import Path

import pytest

from shorts_factory.upload.scheduler import (
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
