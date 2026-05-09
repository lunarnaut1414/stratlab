"""NPR archive scraper.

Walks ``npr.org/sections/<topic>/archive?date=<YYYY-MM-DD>`` one day at a time
and pulls each article's title / author(s) / content / URL into a per-topic
per-year JSON file under ``data/news/npr/<topic>/<year>.json``.

Resumable — articles already on disk are skipped on subsequent runs. Polite
sleep (default 2-3s) between requests; tune via ``--sleep``.

Usage::

    python -m stratlab.news.npr                                  # last 7 days, all topics
    python -m stratlab.news.npr --topics economy technology     # subset
    python -m stratlab.news.npr --start 2024-01-01 --end 2024-12-31  # date range
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from stratlab.data.provider import MARKET_DIR

# News data lives next to market data. MARKET_DIR is the project's data root.
NEWS_DIR: Path = MARKET_DIR.parent / "news"
NPR_DIR: Path = NEWS_DIR / "npr"
LEGACY_NPR_DIR: Path = MARKET_DIR.parent.parent / "data" / "npr"  # ../data/npr if MARKET_DIR is data/market

TOPICS: tuple[str, ...] = (
    "politics", "economy", "science", "world", "news",
    "business", "technology", "culture",
)

ARCHIVE_URL = "https://www.npr.org/sections/{topic}/archive?date={date}"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"

_SENTENCE_SPACE_RE = re.compile(r"([.!?])(?=[^ \n])")


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

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields_of(cls))


def fields_of(cls):  # tiny shim so we don't need dataclasses.fields import here
    import dataclasses
    return dataclasses.fields(cls)


def _normalize_content(text: str) -> str:
    text = text.replace("\n", "").replace("\\'", "'")
    return _SENTENCE_SPACE_RE.sub(r"\1 ", text)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _sleep(base: float, jitter: float = 1.0) -> None:
    if base <= 0:
        return
    import random
    time.sleep(base + random.random() * jitter)


def _archive_articles(soup: BeautifulSoup, topic: str, target_date: date) -> list[str]:
    """Return article URLs from one archive-page soup that match ``target_date``."""
    out: list[str] = []
    target_str = target_date.strftime("%Y/%m/%d")
    for art in soup.find_all("article", attrs={"class": "item"}):
        link = art.find("a")
        if not link:
            continue
        href = link.get("href") or ""
        if target_str not in href:
            continue
        out.append(href)
    return out


def _parse_article(url: str, html: str, topic: str) -> Article | None:
    """Parse an NPR article page into an :class:`Article`. Returns None if
    the page doesn't have parseable content (some links are radio shows, ads, etc).
    """
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Modern NPR: authors are in .byline__name elements (article-level).
    authors = [el.get_text(strip=True) for el in soup.select(".byline__name")]
    # Dedupe while preserving order.
    seen: dict[str, None] = {}
    authors = [a for a in authors if a and not (a in seen or seen.setdefault(a, None))]

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

    # ID derived from URL: /YYYY/MM/DD/<id>/...
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


def _topic_year_path(topic: str, year: int) -> Path:
    return NPR_DIR / topic / f"{year}.json"


def _load_topic_year(topic: str, year: int) -> dict[str, dict]:
    path = _topic_year_path(topic, year)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_topic_year(topic: str, year: int, payload: dict[str, dict]) -> None:
    path = _topic_year_path(topic, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


@dataclass
class NPRScrapeStats:
    fetched_articles: int = 0
    skipped_articles: int = 0
    days_visited: int = 0
    days_with_no_articles: int = 0
    errors: int = 0
    by_topic: dict[str, int] = field(default_factory=dict)


def migrate_legacy_npr() -> int:
    """Move pre-existing ``data/npr/`` payloads into ``data/news/npr/``.

    Renames ``npr-<topic>-article-<year>.json`` to ``<year>.json`` along the
    way. Idempotent — once the source is empty, the function no-ops. Returns
    the count of files moved.
    """
    if not LEGACY_NPR_DIR.exists() or LEGACY_NPR_DIR.resolve() == NPR_DIR.resolve():
        return 0

    moved = 0
    rename_re = re.compile(r"^npr-(?P<topic>[^-]+)-article-(?P<year>\d{4})\.json$")
    for src in LEGACY_NPR_DIR.rglob("*.json"):
        m = rename_re.match(src.name)
        if m:
            topic = m.group("topic")
            year = m.group("year")
            dest_name = f"{year}.json"
        else:
            topic = src.parent.name
            dest_name = src.name
        dest = NPR_DIR / topic / dest_name
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    # Try to remove now-empty source dirs (best effort)
    for sub in sorted(LEGACY_NPR_DIR.glob("*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()
            except OSError:
                pass
    try:
        LEGACY_NPR_DIR.rmdir()
    except OSError:
        pass

    return moved


def scrape(
    topics: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    sleep: float = 2.0,
    verbose: bool = True,
) -> NPRScrapeStats:
    """Scrape NPR articles for ``topics`` over ``[start, end]`` inclusive.

    Defaults: all topics, the last 7 days. ``sleep`` is the base seconds between
    requests (a uniform 0–1 jitter is added on top to look polite). Articles
    already saved locally are skipped, so re-runs only fetch what's missing.
    """
    topics = topics or list(TOPICS)
    today = date.today()
    end = end or today
    start = start or (today - timedelta(days=7))
    if start > end:
        start, end = end, start

    stats = NPRScrapeStats()
    sess = _session()

    # Cache loaded year-files per topic so we don't re-read for every article.
    loaded: dict[tuple[str, int], dict[str, dict]] = {}

    def get_year(topic: str, year: int) -> dict[str, dict]:
        key = (topic, year)
        if key not in loaded:
            loaded[key] = _load_topic_year(topic, year)
        return loaded[key]

    try:
        for topic in topics:
            day = start
            while day <= end:
                stats.days_visited += 1
                day_str = day.strftime("%Y-%m-%d")
                try:
                    _sleep(sleep)
                    r = sess.get(ARCHIVE_URL.format(topic=topic, date=day_str), timeout=30)
                    r.raise_for_status()
                except Exception as exc:
                    if verbose:
                        print(f"  [archive fail] {topic} {day_str}: {exc}")
                    stats.errors += 1
                    day += timedelta(days=1)
                    continue

                soup = BeautifulSoup(r.content, "html.parser")
                links = _archive_articles(soup, topic, day)
                if not links:
                    stats.days_with_no_articles += 1
                    day += timedelta(days=1)
                    continue

                year_payload = get_year(topic, day.year)
                year_dirty = False
                for url in links:
                    parts = [p for p in url.split("/") if p]
                    article_id = None
                    for i in range(len(parts) - 3):
                        if parts[i].isdigit() and len(parts[i]) == 4:
                            article_id = f"{parts[i]}-{parts[i+1]}-{parts[i+2]}-{parts[i+3]}"
                            break
                    if article_id and article_id in year_payload:
                        stats.skipped_articles += 1
                        continue
                    try:
                        _sleep(sleep)
                        ar = sess.get(url, timeout=30)
                        ar.raise_for_status()
                    except Exception as exc:
                        if verbose:
                            print(f"  [article fail] {url}: {exc}")
                        stats.errors += 1
                        continue
                    article = _parse_article(url, ar.text, topic)
                    if article is None:
                        continue
                    year_payload[article.id] = asdict(article)
                    year_dirty = True
                    stats.fetched_articles += 1
                    stats.by_topic[topic] = stats.by_topic.get(topic, 0) + 1
                    if verbose:
                        print(f"  + {article.id} {article.title[:80]}")

                if year_dirty:
                    _save_topic_year(topic, day.year, year_payload)
                day += timedelta(days=1)
    finally:
        # Defensive flush of any in-memory year buffers (no-op if nothing dirty).
        for (topic, year), payload in loaded.items():
            on_disk = _load_topic_year(topic, year)
            if len(payload) > len(on_disk):
                _save_topic_year(topic, year, payload)

    return stats


def _print_summary(stats: NPRScrapeStats, start: date, end: date) -> None:
    print("\n" + "=" * 60)
    print(f"NPR scrape summary  ({start} → {end})")
    print(f"  days visited        : {stats.days_visited}")
    print(f"  empty days          : {stats.days_with_no_articles}")
    print(f"  articles fetched    : {stats.fetched_articles}")
    print(f"  articles skipped    : {stats.skipped_articles}")
    print(f"  errors              : {stats.errors}")
    if stats.by_topic:
        print("  by topic            :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to            : {NPR_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--topics",
        nargs="*",
        choices=TOPICS,
        help=f"Topics to scrape (default: all). Choices: {', '.join(TOPICS)}.",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Earliest date (default: 7 days ago).",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="Latest date (default: today).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Base seconds between requests (default 2; 0–1s jitter is added).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-article progress output.",
    )
    args = parser.parse_args()

    migrated = migrate_legacy_npr()
    if migrated:
        print(f"Migrated {migrated} legacy NPR file(s) into {NPR_DIR}")

    today = date.today()
    end = args.end or today
    start = args.start or (today - timedelta(days=7))

    stats = scrape(
        topics=args.topics,
        start=start,
        end=end,
        sleep=args.sleep,
        verbose=not args.quiet,
    )

    _print_summary(stats, start, end)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
