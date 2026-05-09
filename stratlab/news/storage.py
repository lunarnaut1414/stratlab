"""Per-day JSON storage for news articles.

Layout::

    data/news/<source>/<topic>/<YYYY>/<source>-<topic>-<YYYY-MM-DD>.json

Filenames are intentionally self-describing — the source, topic, and date
appear in the filename itself so a JSON copied or shared out of context
(emailed, dropped in another folder) is still unambiguous about what it is.

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
import re
from datetime import date
from pathlib import Path

from stratlab.data.provider import MARKET_DIR

NEWS_DIR: Path = MARKET_DIR.parent / "news"

_DATE_ONLY_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def day_filename(source: str, topic: str, day: date) -> str:
    return f"{source}-{topic}-{day.isoformat()}.json"


def day_path(source: str, topic: str, day: date) -> Path:
    """Return the absolute path for ``(source, topic, day)``."""
    return NEWS_DIR / source / topic / f"{day.year:04d}" / day_filename(source, topic, day)


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


def migrate_filenames(source: str, verbose: bool = True) -> int:
    """Rename legacy ``<YYYY-MM-DD>.json`` files to ``<source>-<topic>-<date>.json``.

    Walks ``data/news/<source>/<topic>/<YYYY>/`` directories and renames any
    file matching the pre-rename pattern. Idempotent — files already in the
    new format are skipped. Returns the number of files renamed.
    """
    source_root = NEWS_DIR / source
    if not source_root.exists():
        return 0
    renamed = 0
    for topic_dir in source_root.iterdir():
        if not topic_dir.is_dir():
            continue
        topic = topic_dir.name
        for year_dir in topic_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for f in year_dir.glob("*.json"):
                m = _DATE_ONLY_FILE_RE.match(f.name)
                if not m:
                    continue
                new_name = f"{source}-{topic}-{m.group(1)}.json"
                target = year_dir / new_name
                if target.exists():
                    # Conflict (shouldn't happen since same day) — skip
                    continue
                f.rename(target)
                renamed += 1
    if verbose and renamed:
        print(f"Renamed {renamed:,} {source} day-files to <source>-<topic>-<date> format")
    return renamed
