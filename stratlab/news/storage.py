"""Per-day JSON storage for news articles.

Layout::

    data/news/<source>/<topic>/<YYYY>/<YYYY-MM-DD>.json

Each file holds a dict keyed by article ID, value being the article record.
A file existing on disk means "we've checked this day"; an empty ``{}`` is
still a valid (and meaningful) result — it means we visited the archive page
and the topic didn't publish that day.

Writes are atomic via a temp-then-rename pattern, so a crash mid-day leaves
no partial state and the next run will redo the day.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from stratlab.data.provider import MARKET_DIR

NEWS_DIR: Path = MARKET_DIR.parent / "news"


def day_path(source: str, topic: str, day: date) -> Path:
    """Return the absolute path for ``(source, topic, day)``."""
    return NEWS_DIR / source / topic / f"{day.year:04d}" / f"{day.isoformat()}.json"


def day_exists(source: str, topic: str, day: date) -> bool:
    return day_path(source, topic, day).exists()


def load_day(source: str, topic: str, day: date) -> dict[str, dict] | None:
    p = day_path(source, topic, day)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_day(
    source: str,
    topic: str,
    day: date,
    articles: dict[str, dict],
) -> None:
    """Write ``articles`` to the day-file atomically.

    Empty ``articles`` is allowed and meaningful: marks the day as visited
    so future runs skip without an HTTP request.
    """
    p = day_path(source, topic, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(articles, indent=2, sort_keys=True))
    os.replace(tmp, p)  # atomic on POSIX


def iter_days(source: str, topic: str) -> list[Path]:
    """All day-files for ``(source, topic)``, sorted oldest → newest."""
    base = NEWS_DIR / source / topic
    if not base.exists():
        return []
    return sorted(base.rglob("*.json"))
