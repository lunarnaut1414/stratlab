"""BBC News scraper.

Pulls articles from BBC's per-topic RSS feeds, fetches each article's full
HTML, and saves to per-day JSON. RSS feeds give roughly the last 50 articles
per topic — enough for daily-incremental scraping. Deep historical backfill
isn't supported (BBC doesn't expose date-bucketed archives), so this scraper
is best run on a daily cadence to accumulate coverage over time.
"""
from __future__ import annotations

import argparse
import json as _json
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup

from stratlab.news.storage import NEWS_DIR, load_day, save_day

SOURCE = "bbc"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
BBC_DIR = NEWS_DIR / SOURCE

# Storage topic name → BBC RSS slug
TOPICS: dict[str, str] = {
    "business": "business",
    "world": "world",
    "technology": "technology",
    "science": "science_and_environment",
    "politics": "politics",
    "entertainment": "entertainment_and_arts",
    "health": "health",
    "uk": "uk",
}
RSS_URL = "https://feeds.bbci.co.uk/news/{slug}/rss.xml"


@dataclass
class Article:
    id: str
    url: str
    title: str
    authors: list[str]
    published_date: str
    section: str
    content: str
    description: str = ""


@dataclass
class BBCStats:
    feeds_visited: int = 0
    days_updated: int = 0
    fetched_articles: int = 0
    skipped_already_have: int = 0
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


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _sleep(base: float, jitter: float = 1.0) -> None:
    if base <= 0:
        return
    time.sleep(base + random.random() * jitter)


def _extract_article_id(url: str) -> str | None:
    """Pull the slug at the end of a BBC article URL.

    BBC formats: ``/news/articles/<id>``, ``/news/<topic>-<id>``,
    ``/news/world-asia-<id>``. Returns the trailing alphanumeric segment.
    """
    parts = [p for p in url.split("?")[0].split("/") if p]
    if not parts:
        return None
    last = parts[-1]
    if re.fullmatch(r"[a-z0-9]{8,}", last):
        return last
    # Hyphenated form: world-asia-12345678
    m = re.search(r"-(\d{6,})$", last)
    return m.group(1) if m else None


def _fetch_rss(topic: str, session: requests.Session) -> list[tuple[str, str, date | None]]:
    slug = TOPICS[topic]
    r = session.get(RSS_URL.format(slug=slug), timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out: list[tuple[str, str, date | None]] = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()
        pubdate_raw = (item.findtext("pubDate") or "").strip()
        pub_date: date | None = None
        if pubdate_raw:
            try:
                pub_date = parsedate_to_datetime(pubdate_raw).date()
            except Exception:
                pass
        if link:
            out.append((link, title, pub_date))
    return out


def _parse_article(url: str, html: str, topic: str, fallback_date: date | None) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    authors: list[str] = []
    pub_date_iso = ""
    description = ""
    for ld in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(ld.string)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("@type") in ("NewsArticle", "Article", "ReportageNewsArticle"):
            au = data.get("author")
            if isinstance(au, list):
                for a in au:
                    if isinstance(a, dict) and a.get("name"):
                        authors.append(a["name"])
            elif isinstance(au, dict) and au.get("name"):
                authors.append(au["name"])
            pub_date_iso = data.get("datePublished") or pub_date_iso
            description = data.get("description") or description
            break

    paragraphs: list[str] = []
    for block in soup.select('[data-component="text-block"]'):
        for p in block.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
    if not paragraphs:
        # Fallback: <article> > <p> with reasonable length
        article_tag = soup.find("article")
        if article_tag:
            for p in article_tag.find_all("p"):
                text = p.get_text(" ", strip=True)
                if text and len(text) > 30:
                    paragraphs.append(text)
    if not paragraphs:
        return None

    pub_date = pub_date_iso[:10] if pub_date_iso else (fallback_date.isoformat() if fallback_date else "")
    article_id = _extract_article_id(url)
    if not article_id:
        return None

    return Article(
        id=f"{pub_date}-{article_id}" if pub_date else article_id,
        url=url,
        title=title,
        authors=authors,
        published_date=pub_date,
        section=topic,
        content=" ".join(paragraphs),
        description=description,
    )


def _scrape_topic(
    topic: str,
    session: requests.Session,
    sleep_base: float,
    stats: BBCStats,
    verbose: bool,
) -> None:
    try:
        items = _fetch_rss(topic, session)
    except Exception as exc:
        if verbose:
            print(f"  [rss fail] {topic}: {exc}")
        stats.add(errors=1)
        return
    stats.add(feeds_visited=1)

    # Cache day-files we've touched this run so we don't re-read disk per article.
    day_cache: dict[date, dict[str, dict]] = {}
    days_with_new: set[date] = set()

    for url, title, pub_date in items:
        if pub_date is None:
            continue
        article_id_raw = _extract_article_id(url)
        if not article_id_raw:
            continue
        article_key = f"{pub_date.isoformat()}-{article_id_raw}"

        if pub_date not in day_cache:
            day_cache[pub_date] = load_day(SOURCE, topic, pub_date) or {}
        if article_key in day_cache[pub_date]:
            stats.add(skipped_already_have=1)
            continue

        try:
            _sleep(sleep_base)
            ar = session.get(url, timeout=30)
            ar.raise_for_status()
        except Exception as exc:
            if verbose:
                print(f"  [article fail] {url}: {exc}")
            stats.add(errors=1)
            continue
        article = _parse_article(url, ar.text, topic, fallback_date=pub_date)
        if not article:
            continue
        day_cache[pub_date][article.id] = asdict(article)
        days_with_new.add(pub_date)
        stats.add(fetched_articles=1, by_topic={topic: 1})

    for day in days_with_new:
        save_day(SOURCE, topic, day, day_cache[day])
        stats.add(days_updated=1)
        if verbose:
            print(f"  + {topic} {day}: {len(day_cache[day])} total")


def scrape(
    topics: list[str] | None = None,
    sleep: float = 1.0,
    workers: int = 1,
    verbose: bool = True,
) -> BBCStats:
    """Scrape BBC RSS feeds for ``topics`` and fetch each fresh article."""
    topics = topics or list(TOPICS.keys())
    stats = BBCStats()

    if workers <= 1:
        session = _new_session()
        for topic in topics:
            _scrape_topic(topic, session, sleep, stats, verbose)
    else:
        def run(topic: str) -> None:
            _scrape_topic(topic, _new_session(), sleep, stats, verbose)
        with ThreadPoolExecutor(max_workers=min(workers, len(topics))) as ex:
            futures = {ex.submit(run, t): t for t in topics}
            for fut in as_completed(futures):
                topic = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    if verbose:
                        print(f"  [worker {topic} crashed] {exc}")
                    stats.add(errors=1)

    return stats


def _print_summary(stats: BBCStats) -> None:
    print("\n" + "=" * 60)
    print("BBC scrape summary")
    print(f"  feeds visited           : {stats.feeds_visited}")
    print(f"  days updated            : {stats.days_updated}")
    print(f"  articles fetched        : {stats.fetched_articles}")
    print(f"  skipped (already have)  : {stats.skipped_already_have}")
    print(f"  errors                  : {stats.errors}")
    if stats.by_topic:
        print("  by topic                :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to                : {BBC_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topics", nargs="*", choices=list(TOPICS.keys()),
                        help=f"Topics (default: all). Choices: {', '.join(TOPICS.keys())}.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Base seconds between requests (+0-1s jitter).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel topic workers (1 by default; 4-8 for backfill).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stats = scrape(
        topics=args.topics,
        sleep=args.sleep,
        workers=args.workers,
        verbose=not args.quiet,
    )
    _print_summary(stats)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
