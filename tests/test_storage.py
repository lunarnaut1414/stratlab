"""Per-day news storage primitives.

Atomic write + crash safety is what makes long news backfills resumable.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stratlab.news import storage


def _redirect_news_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(storage, "NEWS_DIR", tmp_path)


def test_save_load_round_trip(tmp_path, monkeypatch):
    _redirect_news_dir(monkeypatch, tmp_path)
    day = date(2024, 5, 1)
    payload = {"id1": {"title": "x", "content": "y"}}
    storage.save_day("npr", "business", day, payload)
    loaded = storage.load_day("npr", "business", day)
    assert loaded == payload


def test_load_day_returns_none_when_missing(tmp_path, monkeypatch):
    _redirect_news_dir(monkeypatch, tmp_path)
    assert storage.load_day("npr", "missing", date(2024, 1, 1)) is None
    assert not storage.day_exists("npr", "missing", date(2024, 1, 1))


def test_save_empty_dict_is_meaningful(tmp_path, monkeypatch):
    """An empty dict still creates the day-file — used to mark 'we visited
    this day, the topic was empty'. Future runs should skip without an HTTP."""
    _redirect_news_dir(monkeypatch, tmp_path)
    day = date(2024, 5, 2)
    storage.save_day("npr", "business", day, {})
    assert storage.day_exists("npr", "business", day)
    assert storage.load_day("npr", "business", day) == {}


def test_save_uses_atomic_rename_no_tmp_left_behind(tmp_path, monkeypatch):
    """After save_day completes, the .tmp sibling must not exist on disk."""
    _redirect_news_dir(monkeypatch, tmp_path)
    day = date(2024, 5, 3)
    storage.save_day("npr", "business", day, {"id": {"title": "x"}})
    target = storage.day_path("npr", "business", day)
    assert target.exists()
    tmp = target.with_suffix(".json.tmp")
    assert not tmp.exists(), "atomic rename left .tmp file behind"


def test_filename_layout_includes_source_topic_date(tmp_path, monkeypatch):
    """Filename must be self-describing: <source>-<topic>-<YYYY-MM-DD>.json
    so a JSON copied out of context isn't ambiguous."""
    _redirect_news_dir(monkeypatch, tmp_path)
    day = date(2024, 5, 4)
    storage.save_day("ap", "world", day, {})
    p = storage.day_path("ap", "world", day)
    assert p.name == "ap-world-2024-05-04.json"
    # Path also includes <source>/<topic>/<year>/...
    assert p.relative_to(tmp_path).parts == (
        "ap", "world", "2024", "ap-world-2024-05-04.json",
    )


def test_load_day_handles_corrupt_json(tmp_path, monkeypatch):
    """Corrupt JSON shouldn't crash the loader — must return None so the next
    refresh redoes the day."""
    _redirect_news_dir(monkeypatch, tmp_path)
    day = date(2024, 5, 5)
    p = storage.day_path("npr", "business", day)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json")
    assert storage.load_day("npr", "business", day) is None
