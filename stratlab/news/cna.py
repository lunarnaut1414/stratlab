"""Channel News Asia (Singapore) scraper.

CNA exposes a clean ``/api/v1/sitemap-news-feed`` endpoint with the 50 most
recent articles, including ``news:publication_date`` and topic-bearing URLs
(``/asia/``, ``/singapore/``, ``/business/``, etc). Their robots.txt sets
``Crawl-delay: 10`` which we respect by defaulting ``--sleep 2`` (still
polite for a major regional outlet).

NHK was considered but their robots.txt explicitly disallows AI/scraper
bots (``User-agent: anthropic-ai`` ``Disallow: /``) so we go with CNA
instead — equivalent profile (English-language Asian flagship) and
explicitly scrape-friendly.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from stratlab.news.storage import NEWS_DIR, load_day, save_day

SOURCE = "cna"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
CNA_DIR = NEWS_DIR / SOURCE
CNA_BASE = "https://www.channelnewsasia.com"
SITEMAP_NEWS_FEED = f"{CNA_BASE}/api/v1/sitemap-news-feed"

# CNA's URL path → our topic vocabulary. Paths not in this set fall through
# to GENERAL_TOPIC. CNA also uses /commentary, /lifestyle, /sustainability,
# /east-asia — extend the map as you find more in the wild.
GENERAL_TOPIC = "general"
_PATH_TO_TOPIC: dict[str, str] = {
    "asia": "asia",
    "east-asia": "asia",
    "singapore": "singapore",
    "world": "world",
    "business": "business",
    "sport": "sport",
    "commentary": "commentary",
    "lifestyle": "lifestyle",
    "sustainability": "sustainability",
    "health": "health",
    "wellness": "health",
    "tech": "technology",
    "technology": "technology",
    "entertainment": "entertainment",
    "cnainsider": "feature",
    "cna-lifestyle": "lifestyle",
}
TOPICS: tuple[str, ...] = tuple(sorted(set(_PATH_TO_TOPIC.values()) | {GENERAL_TOPIC}))

_ARTICLE_ID_RE = re.compile(r"-(\d{6,})/?$")


@dataclass
class Article:
    id: str
    url: str
    title: str
    authors: list[str]
    published_date: str
    section: str
    content: str
    keywords: str = ""
    description: str = ""


@dataclass
class CNAStats:
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


def _topic_from_url(url: str) -> str:
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts:
        return GENERAL_TOPIC
    return _PATH_TO_TOPIC.get(parts[0], GENERAL_TOPIC)


def _extract_article_id(url: str) -> str | None:
    m = _ARTICLE_ID_RE.search(urlparse(url).path)
    return m.group(1) if m else None


def _parse_xml(content: bytes) -> ET.Element:
    root = ET.fromstring(content)
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _fetch_sitemap_feed(session: requests.Session) -> list[dict]:
    """Return list of ``{url, pub_date, title, keywords}`` dicts."""
    r = session.get(SITEMAP_NEWS_FEED, timeout=30)
    r.raise_for_status()
    root = _parse_xml(r.content)
    out: list[dict] = []
    for url_el in root.iter("url"):
        loc = url_el.findtext("loc") or ""
        if not loc:
            continue
        news = url_el.find("news")
        pub_date_str = ""
        title = ""
        keywords = ""
        if news is not None:
            pub_date_str = news.findtext("publication_date") or ""
            title = news.findtext("title") or ""
            keywords = news.findtext("keywords") or ""
        pub_date: date | None = None
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str).date()
            except ValueError:
                pass
        out.append({
            "url": loc.strip(),
            "pub_date": pub_date,
            "title": title.strip(),
            "keywords": keywords.strip(),
        })
    return out


def _parse_article(url: str, html: str, fallback_date: date | None,
                   sitemap_title: str = "", sitemap_keywords: str = "") -> Article | None:
    soup = BeautifulSoup(html, "html.parser")

    title = sitemap_title or (soup.title.string.strip() if soup.title and soup.title.string else url)

    authors: list[str] = []
    for meta in soup.find_all("meta", attrs={"name": "cXenseParse:author"}):
        name = (meta.get("content") or "").strip()
        if name and name not in authors:
            authors.append(name)

    description = ""
    desc_meta = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
        "meta", attrs={"name": "description"}
    )
    if desc_meta and desc_meta.get("content"):
        description = desc_meta["content"].strip()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("div.text-long p")]
    paragraphs = [p for p in paragraphs if p and len(p) > 20]
    if not paragraphs:
        # Fallback: any <p> within the main article tag
        article_tag = soup.find("article") or soup.find("main")
        if article_tag:
            paragraphs = [
                p.get_text(" ", strip=True) for p in article_tag.find_all("p")
                if p.get_text(strip=True) and len(p.get_text(strip=True)) > 30
            ]
    if not paragraphs:
        return None

    article_id = _extract_article_id(url)
    if not article_id:
        return None
    pub_date = fallback_date.isoformat() if fallback_date else ""
    topic = _topic_from_url(url)

    return Article(
        id=f"{pub_date}-{article_id}" if pub_date else article_id,
        url=url,
        title=title,
        authors=authors,
        published_date=pub_date,
        section=topic,
        content=" ".join(paragraphs),
        keywords=sitemap_keywords,
        description=description,
    )


def _scrape_chunk(items: list[dict], sleep_base: float, stats: CNAStats, verbose: bool) -> None:
    session = _new_session()
    day_cache: dict[tuple[str, date], dict[str, dict]] = {}
    dirty: set[tuple[str, date]] = set()

    for item in items:
        url = item["url"]
        pub_date: date | None = item["pub_date"]
        if pub_date is None:
            continue
        topic = _topic_from_url(url)
        article_id_raw = _extract_article_id(url)
        if not article_id_raw:
            continue
        cache_key = (topic, pub_date)
        if cache_key not in day_cache:
            day_cache[cache_key] = load_day(SOURCE, topic, pub_date) or {}
        article_key = f"{pub_date.isoformat()}-{article_id_raw}"
        if article_key in day_cache[cache_key]:
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
        article = _parse_article(
            url, ar.text, fallback_date=pub_date,
            sitemap_title=item.get("title", ""),
            sitemap_keywords=item.get("keywords", ""),
        )
        if not article:
            continue
        day_cache[cache_key][article.id] = asdict(article)
        dirty.add(cache_key)
        stats.add(fetched_articles=1, by_topic={topic: 1})

    for (topic, day) in dirty:
        save_day(SOURCE, topic, day, day_cache[(topic, day)])
        stats.add(days_updated=1)


def _chunk(items: list, n: int) -> list[list]:
    if n <= 1 or len(items) <= n:
        return [items] if items else []
    out = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return [c for c in out if c]


def scrape(
    topics: list[str] | None = None,
    sleep: float = 2.0,
    workers: int = 1,
    verbose: bool = True,
) -> CNAStats:
    """Scrape CNA's sitemap-news-feed for the latest 50 articles.

    ``topics`` filters by URL-derived topic post-discovery (the feed itself
    isn't topic-segmented). ``--workers > 1`` chunks the candidate list
    across worker threads.
    """
    stats = CNAStats()
    discovery_session = _new_session()

    try:
        feed_items = _fetch_sitemap_feed(discovery_session)
        stats.add(feeds_visited=1)
    except Exception as exc:
        if verbose:
            print(f"  [feed fail]: {exc}")
        stats.add(errors=1)
        return stats

    if topics:
        keep = set(topics)
        feed_items = [it for it in feed_items if _topic_from_url(it["url"]) in keep]

    if verbose:
        print(f"CNA feed: {len(feed_items)} candidate article(s)")

    chunks = _chunk(feed_items, max(1, workers))
    if workers <= 1 or len(chunks) == 1:
        for chunk in chunks:
            _scrape_chunk(chunk, sleep, stats, verbose)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_scrape_chunk, c, sleep, stats, verbose) for c in chunks]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    if verbose:
                        print(f"  [worker crashed] {exc}")
                    stats.add(errors=1)

    return stats


def _print_summary(stats: CNAStats) -> None:
    print("\n" + "=" * 60)
    print("CNA scrape summary")
    print(f"  feeds visited           : {stats.feeds_visited}")
    print(f"  days updated            : {stats.days_updated}")
    print(f"  articles fetched        : {stats.fetched_articles}")
    print(f"  skipped (already have)  : {stats.skipped_already_have}")
    print(f"  errors                  : {stats.errors}")
    if stats.by_topic:
        print("  by topic                :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to                : {CNA_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topics", nargs="*", choices=TOPICS,
                        help=f"Topics (default: all). Choices: {', '.join(TOPICS)}.")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="Base seconds between requests "
                             "(robots.txt requests Crawl-delay: 10; default 2 with jitter).")
    parser.add_argument("--workers", type=int, default=1)
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
