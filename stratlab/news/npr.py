"""NPR archive scraper, per-day storage.

Walks ``npr.org/sections/<topic>/archive?date=<YYYY-MM-DD>`` one day at a time,
extracts each article's title / authors / content / URL, and saves the day's
articles to ``data/news/npr/<topic>/<YYYY>/<YYYY-MM-DD>.json``.

Day-level resume: if the day file already exists locally, the entire day is
skipped (zero HTTP requests). Empty days still get a ``{}`` file written so
they don't get re-checked. Atomic writes ensure a crash mid-day leaves no
partial state — the next run redoes that day cleanly.

Usage::

    python -m stratlab.news.npr                       # last 7 days, all topics
    python -m stratlab.news.npr --full                # all of 2000→today
    python -m stratlab.news.npr --start 2024-01-01    # explicit window
    python -m stratlab.news.npr --workers 4           # parallelize across topics
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from stratlab.news.storage import (
    NEWS_DIR,
    day_exists,
    day_path,
    load_day,
    migrate_filenames,
    save_day,
)

NPR_DIR: Path = NEWS_DIR / "npr"
LEGACY_BACKUP_DIR: Path = NEWS_DIR / "_legacy_yearly_backup" / "npr"

TOPICS: tuple[str, ...] = (
    "politics", "economy", "science", "world", "news",
    "business", "technology", "culture",
)

ARCHIVE_URL = "https://www.npr.org/sections/{topic}/archive?date={date}"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
SOURCE = "npr"
FULL_BACKFILL_START = date(2000, 1, 1)

_SENTENCE_SPACE_RE = re.compile(r"([.!?])(?=[^ \n])")
_LEGACY_NAME_RE = re.compile(r"^(?P<topic>[^/]+?)/(?P<year>\d{4})\.json$")


@dataclass
class Article:
    id: str
    url: str
    title: str
    authors: list[str]
    published_date: str
    section: str
    content: str
    disclaimer: str = ""


@dataclass
class NPRScrapeStats:
    days_visited: int = 0
    days_skipped_cached: int = 0
    days_with_no_articles: int = 0
    fetched_articles: int = 0
    errors: int = 0
    by_topic: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **kwargs) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if k == "by_topic":
                    for topic, n in v.items():
                        self.by_topic[topic] = self.by_topic.get(topic, 0) + n
                else:
                    setattr(self, k, getattr(self, k) + v)


# ---------------------------------------------------------------------------
# HTTP / parsing helpers
# ---------------------------------------------------------------------------

def _normalize_content(text: str) -> str:
    text = text.replace("\n", "").replace("\\'", "'")
    return _SENTENCE_SPACE_RE.sub(r"\1 ", text)


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _sleep(base: float, jitter: float = 1.0) -> None:
    if base <= 0:
        return
    time.sleep(base + random.random() * jitter)


def _archive_articles(soup: BeautifulSoup, target_date: date) -> list[str]:
    target_str = target_date.strftime("%Y/%m/%d")
    out: list[str] = []
    for art in soup.find_all("article", attrs={"class": "item"}):
        link = art.find("a")
        if not link:
            continue
        href = link.get("href") or ""
        if target_str in href:
            out.append(href)
    return out


def _parse_article(url: str, html: str, topic: str) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    authors_raw = [el.get_text(strip=True) for el in soup.select(".byline__name")]
    seen: dict[str, None] = {}
    authors = [a for a in authors_raw if a and not (a in seen or seen.setdefault(a, None))]

    content_divs = soup.select("div.storytext")
    if not content_divs:
        return None

    paragraphs: list[str] = []
    disclaimer = ""
    for div in content_divs:
        for el in div.find_all(["p"]):
            classes = el.get("class") or []
            if "disclaimer" in classes:
                disclaimer += " " + el.get_text(" ", strip=True)
                continue
            if "caption" in classes:
                continue
            text = el.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
    if not paragraphs:
        return None
    content = _normalize_content(" ".join(paragraphs))

    parts = [p for p in url.split("/") if p]
    article_id = ""
    pub_date = ""
    for i in range(len(parts) - 3):
        if parts[i].isdigit() and len(parts[i]) == 4 and parts[i + 1].isdigit() and parts[i + 2].isdigit():
            pub_date = f"{parts[i]}-{parts[i+1]}-{parts[i+2]}"
            article_id = f"{pub_date}-{parts[i+3]}"
            break
    if not article_id:
        return None

    return Article(
        id=article_id,
        url=url,
        title=title,
        authors=authors,
        published_date=pub_date,
        section=topic,
        content=content,
        disclaimer=disclaimer.strip(),
    )


# ---------------------------------------------------------------------------
# Migration: yearly → per-day, with backup
# ---------------------------------------------------------------------------

def migrate_yearly_to_daily(verbose: bool = True) -> dict[str, int]:
    """Split legacy ``<topic>/<year>.json`` files into per-day files.

    Yearly originals are moved to ``data/news/_legacy_yearly_backup/npr/`` so
    nothing is lost. Also catches up the per-day filename format if needed
    (legacy ``<YYYY-MM-DD>.json`` → ``npr-<topic>-<YYYY-MM-DD>.json``).
    Idempotent — once everything is current, subsequent calls no-op.
    """
    # Catch up filenames first so any subsequent operations see the new layout.
    migrate_filenames(SOURCE, verbose=verbose)

    if not NPR_DIR.exists():
        return {"yearly_files": 0, "days_written": 0, "articles": 0}

    # Find legacy yearly files: data/news/npr/<topic>/<year>.json (4-digit name).
    yearly_files: list[Path] = []
    for sub in NPR_DIR.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("*.json"):
            if re.fullmatch(r"\d{4}\.json", f.name):
                yearly_files.append(f)

    if not yearly_files:
        return {"yearly_files": 0, "days_written": 0, "articles": 0}

    if verbose:
        print(f"Migrating {len(yearly_files)} legacy yearly file(s) to per-day layout...")

    days_written = 0
    articles_total = 0
    for yearly in sorted(yearly_files):
        topic = yearly.parent.name
        try:
            payload = json.loads(yearly.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            if verbose:
                print(f"  skip {yearly}: {exc}")
            continue

        # Group articles by published_date
        by_day: dict[str, dict[str, dict]] = {}
        for aid, art in payload.items():
            pub = art.get("published_date") if isinstance(art, dict) else None
            if not pub:
                continue
            by_day.setdefault(pub, {})[aid] = art

        for day_str, day_articles in by_day.items():
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue
            existing = load_day(SOURCE, topic, day) or {}
            existing.update(day_articles)
            save_day(SOURCE, topic, day, existing)
            days_written += 1
            articles_total += len(day_articles)

        # Move yearly file to backup
        backup = LEGACY_BACKUP_DIR / topic / yearly.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(yearly), str(backup))

    if verbose:
        print(
            f"  {len(yearly_files)} yearly file(s) split into {days_written:,} day-files "
            f"({articles_total:,} articles); originals in {LEGACY_BACKUP_DIR}"
        )

    return {
        "yearly_files": len(yearly_files),
        "days_written": days_written,
        "articles": articles_total,
    }


# ---------------------------------------------------------------------------
# Scrape one (topic, day)
# ---------------------------------------------------------------------------

def _scrape_topic_day(
    topic: str,
    day: date,
    session: requests.Session,
    sleep: float,
    stats: NPRScrapeStats,
    verbose: bool,
) -> None:
    if day_exists(SOURCE, topic, day):
        stats.add(days_skipped_cached=1)
        return

    stats.add(days_visited=1)
    day_str = day.strftime("%Y-%m-%d")

    try:
        _sleep(sleep)
        r = session.get(ARCHIVE_URL.format(topic=topic, date=day_str), timeout=30)
        r.raise_for_status()
    except Exception as exc:
        if verbose:
            print(f"  [archive fail] {topic} {day_str}: {exc}")
        stats.add(errors=1)
        return

    soup = BeautifulSoup(r.content, "html.parser")
    links = _archive_articles(soup, day)
    if not links:
        # Mark day as visited (empty) so we don't re-check.
        save_day(SOURCE, topic, day, {})
        stats.add(days_with_no_articles=1)
        return

    articles: dict[str, dict] = {}
    for url in links:
        try:
            _sleep(sleep)
            ar = session.get(url, timeout=30)
            ar.raise_for_status()
        except Exception as exc:
            if verbose:
                print(f"  [article fail] {url}: {exc}")
            stats.add(errors=1)
            continue
        article = _parse_article(url, ar.text, topic)
        if article is None:
            continue
        articles[article.id] = asdict(article)

    save_day(SOURCE, topic, day, articles)
    if articles:
        stats.add(fetched_articles=len(articles), by_topic={topic: len(articles)})
        if verbose:
            print(f"  + {topic} {day_str}: {len(articles)} article(s)")


def _scrape_topic(
    topic: str,
    start: date,
    end: date,
    sleep: float,
    stats: NPRScrapeStats,
    verbose: bool,
) -> None:
    """Walk ``[start, end]`` one day at a time for a single topic."""
    session = _new_session()
    day = start
    while day <= end:
        _scrape_topic_day(topic, day, session, sleep, stats, verbose)
        day += timedelta(days=1)


def scrape(
    topics: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    sleep: float = 1.0,
    workers: int = 1,
    verbose: bool = True,
) -> NPRScrapeStats:
    """Scrape NPR articles for ``topics`` over ``[start, end]`` inclusive.

    With ``workers > 1``, topics are walked in parallel — each worker has its
    own session and sleeps independently, so wall-clock backfill scales
    roughly linearly with the worker count up to the topic count (8).
    """
    topics = topics or list(TOPICS)
    today = date.today()
    end = end or today
    start = start or (today - timedelta(days=7))
    if start > end:
        start, end = end, start

    stats = NPRScrapeStats()

    if workers <= 1:
        for topic in topics:
            _scrape_topic(topic, start, end, sleep, stats, verbose)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(topics))) as ex:
            futures = {
                ex.submit(_scrape_topic, t, start, end, sleep, stats, verbose): t
                for t in topics
            }
            for fut in as_completed(futures):
                topic = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    if verbose:
                        print(f"  [worker {topic} crashed] {exc}")
                    stats.add(errors=1)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(stats: NPRScrapeStats, start: date, end: date) -> None:
    print("\n" + "=" * 60)
    print(f"NPR scrape summary  ({start} → {end})")
    print(f"  days visited            : {stats.days_visited}")
    print(f"  days skipped (cached)   : {stats.days_skipped_cached}")
    print(f"  empty days              : {stats.days_with_no_articles}")
    print(f"  articles fetched        : {stats.fetched_articles}")
    print(f"  errors                  : {stats.errors}")
    if stats.by_topic:
        print("  by topic                :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to                : {NPR_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--topics",
        nargs="*",
        choices=TOPICS,
        help=f"Topics (default: all). Choices: {', '.join(TOPICS)}.",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Earliest date (default: 7 days ago, or 2000-01-01 with --full).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Latest date (default: today).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=f"Backfill from {FULL_BACKFILL_START} to today (slow). Overrides --start.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel topic workers (default 1; backfill goes faster with 4-8).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Base seconds between requests per worker (default 1; +0-1s jitter).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-day output.")
    args = parser.parse_args()

    migration = migrate_yearly_to_daily(verbose=not args.quiet)

    today = date.today()
    if args.full:
        start = FULL_BACKFILL_START
    else:
        start = args.start or (today - timedelta(days=7))
    end = args.end or today

    stats = scrape(
        topics=args.topics,
        start=start,
        end=end,
        sleep=args.sleep,
        workers=args.workers,
        verbose=not args.quiet,
    )

    _print_summary(stats, start, end)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
