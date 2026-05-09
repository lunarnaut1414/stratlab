"""AP News scraper.

AP doesn't expose a public RSS feed anymore, so we walk each topic's hub
page (``apnews.com/hub/<slug>``) for recent article URLs and fetch each
article's HTML. Coverage per run is roughly the latest ~80-100 articles
per topic — running daily accumulates a useful corpus over time.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date

import requests
from bs4 import BeautifulSoup

from stratlab.news.storage import NEWS_DIR, load_day, save_day

SOURCE = "ap"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
AP_DIR = NEWS_DIR / SOURCE
AP_BASE = "https://apnews.com"

# Storage topic name → AP hub slug
TOPICS: dict[str, str] = {
    "business": "business",
    "world": "world-news",
    "technology": "technology",
    "politics": "politics",
    "science": "science",
    "health": "health",
    "entertainment": "entertainment",
    "sports": "sports",
    "us-news": "us-news",
}
HUB_URL = "https://apnews.com/hub/{slug}"

# AP article URL: /article/<slug>-<32-or-more-hex>
_AP_ID_RE = re.compile(r"/article/[^/]+?-([a-f0-9]{16,})/?(?:\?|$)")


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
class APStats:
    hubs_visited: int = 0
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
    m = _AP_ID_RE.search(url)
    return m.group(1) if m else None


def _author_from_url(author_url: str) -> str:
    """``apnews.com/author/gerald-imray`` → ``Gerald Imray``."""
    if not author_url:
        return ""
    parts = [p for p in author_url.rstrip("/").split("/") if p]
    if not parts:
        return ""
    slug = parts[-1].split("?")[0]
    return " ".join(w.capitalize() for w in slug.split("-"))


def _fetch_hub(topic: str, session: requests.Session) -> list[str]:
    slug = TOPICS[topic]
    r = session.get(HUB_URL.format(slug=slug), timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    urls: set[str] = set()
    for a in soup.select('a[href*="/article/"]'):
        href = a.get("href")
        if not href:
            continue
        if href.startswith("/"):
            href = AP_BASE + href
        if "/article/" not in href:
            continue
        urls.add(href.split("?")[0])
    return sorted(urls)


def _parse_article(url: str, html: str, topic: str) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    title = title.replace(" | AP News", "").strip()

    pub_date_iso = ""
    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_date and meta_date.get("content"):
        pub_date_iso = meta_date["content"]

    authors: list[str] = []
    for meta_author in soup.find_all("meta", attrs={"property": "article:author"}):
        name = _author_from_url(meta_author.get("content", ""))
        if name and name not in authors:
            authors.append(name)

    description = ""
    meta_desc = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
        "meta", attrs={"name": "description"}
    )
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"]

    body = soup.select_one(".RichTextStoryBody") or soup.find("main")
    paragraphs: list[str] = []
    if body:
        for p in body.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)
    if not paragraphs:
        return None

    article_id = _extract_article_id(url)
    if not article_id:
        return None

    pub_date = pub_date_iso[:10] if pub_date_iso else ""

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
    stats: APStats,
    verbose: bool,
) -> None:
    try:
        urls = _fetch_hub(topic, session)
    except Exception as exc:
        if verbose:
            print(f"  [hub fail] {topic}: {exc}")
        stats.add(errors=1)
        return
    stats.add(hubs_visited=1)

    day_cache: dict[date, dict[str, dict]] = {}
    days_with_new: set[date] = set()

    for url in urls:
        try:
            _sleep(sleep_base)
            ar = session.get(url, timeout=30)
            ar.raise_for_status()
        except Exception as exc:
            if verbose:
                print(f"  [article fail] {url}: {exc}")
            stats.add(errors=1)
            continue

        article = _parse_article(url, ar.text, topic)
        if not article or not article.published_date:
            continue
        try:
            day = date.fromisoformat(article.published_date)
        except ValueError:
            continue

        if day not in day_cache:
            day_cache[day] = load_day(SOURCE, topic, day) or {}
        if article.id in day_cache[day]:
            stats.add(skipped_already_have=1)
            continue
        day_cache[day][article.id] = asdict(article)
        days_with_new.add(day)
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
) -> APStats:
    topics = topics or list(TOPICS.keys())
    stats = APStats()

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


def _print_summary(stats: APStats) -> None:
    print("\n" + "=" * 60)
    print("AP News scrape summary")
    print(f"  hubs visited            : {stats.hubs_visited}")
    print(f"  days updated            : {stats.days_updated}")
    print(f"  articles fetched        : {stats.fetched_articles}")
    print(f"  skipped (already have)  : {stats.skipped_already_have}")
    print(f"  errors                  : {stats.errors}")
    if stats.by_topic:
        print("  by topic                :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to                : {AP_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topics", nargs="*", choices=list(TOPICS.keys()),
                        help=f"Topics (default: all). Choices: {', '.join(TOPICS.keys())}.")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Base seconds between requests (+0-1s jitter).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel topic workers (1 by default; 4 for faster runs).")
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
