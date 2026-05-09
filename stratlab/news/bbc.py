"""BBC News scraper.

Two discovery modes:

* **RSS (default)** — per-topic RSS feeds at ``feeds.bbci.co.uk``. Gives the
  last ~50 articles per topic. Fast, low-volume, ideal for daily runs.
* **Sitemap (``--from-sitemap``)** — BBC's XML sitemap index goes back to
  ~2009 across ~120 child sitemaps. Use this for historical backfill;
  supports ``--years N`` and ``--since YYYY-MM-DD`` for date scoping.

Either mode pipes through the same article fetcher → parser → per-day JSON
save. Articles already on disk are skipped (article-id-level dedupe inside
day files), so a sitemap backfill pass plays nicely with the RSS daily.
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
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

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

# Sitemap discovery — for historical backfill via --from-sitemap.
SITEMAP_NEWS_INDEX = "https://www.bbc.com/sitemaps/https-index-com-news.xml"
SITEMAP_ARCHIVE_INDEX = "https://www.bbc.com/sitemaps/https-index-com-archive.xml"
GENERAL_TOPIC = "general"  # URL didn't reveal a topic

# BBC's `meta name="page.subsection"` value → our topic vocabulary.
# Anything not in this map falls back to GENERAL_TOPIC. The list is
# best-effort; refine as you see real-world subsection values.
_SUBSECTION_TO_TOPIC = {
    "business": "business",
    "world": "world",
    "world news": "world",
    "us & canada": "world",
    "europe": "world",
    "asia": "world",
    "asia-pacific": "world",
    "africa": "world",
    "middle east": "world",
    "latin america": "world",
    "tech": "technology",
    "technology": "technology",
    "science": "science",
    "science & environment": "science",
    "science and environment": "science",
    "environment": "science",
    "health": "health",
    "politics": "politics",
    "entertainment": "entertainment",
    "entertainment & arts": "entertainment",
    "entertainment and arts": "entertainment",
    "arts": "entertainment",
    "uk": "uk",
    "uk politics": "politics",
}


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

    # BBC's `meta page.subsection` is the most reliable section signal on
    # modern article pages. Use it when present; otherwise fall back to
    # whatever the caller passed in (e.g. the RSS feed name).
    section = topic
    sub = soup.find("meta", attrs={"name": "page.subsection"})
    if sub and sub.get("content"):
        section = sub["content"].strip()

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
        section=section,
        content=" ".join(paragraphs),
        description=description,
    )


def _topic_for_storage(article_section: str, url: str) -> str:
    """Final topic decision for routing to a topic folder.

    Priority: known subsection from page.subsection → URL-inferred topic →
    ``GENERAL_TOPIC``.
    """
    if article_section:
        mapped = _SUBSECTION_TO_TOPIC.get(article_section.strip().lower())
        if mapped:
            return mapped
    return _topic_from_url(url)


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


def _topic_from_url(url: str) -> str:
    """Best-effort topic extraction from a BBC article URL path.

    - ``/news/business-12345678`` → ``"business"``
    - ``/news/world-asia-12345678`` → ``"world"``
    - ``/news/uk-england-london-12345678`` → ``"uk"``
    - ``/news/articles/<id>`` (modern format, no topic) → ``GENERAL_TOPIC``
    """
    path = urlparse(url).path
    parts = [p for p in path.split("/") if p]
    if not parts or parts[0] != "news":
        return GENERAL_TOPIC
    if len(parts) >= 2 and parts[1] == "articles":
        return GENERAL_TOPIC
    if len(parts) >= 2:
        slug = parts[1]
        for known in ("business", "world", "technology", "science", "health",
                      "politics", "entertainment", "uk"):
            if slug == known or slug.startswith(known + "-"):
                return known
    return GENERAL_TOPIC


def _parse_sitemap(content: bytes) -> ET.Element:
    """Parse XML and strip default namespace for easier traversal."""
    root = ET.fromstring(content)
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _parse_iso_date(s: str) -> date | None:
    if not s:
        return None
    try:
        # BBC sitemaps use 2026-05-09T08:06:53Z
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _discover_via_sitemap(
    session: requests.Session,
    since: date,
    topics_filter: set[str] | None,
    verbose: bool,
) -> list[tuple[str, date, str]]:
    """Walk BBC's news + archive sitemap indexes and yield article candidates.

    Returns a list of ``(article_url, lastmod_date, inferred_topic)`` tuples
    for English-language ``/news/...`` articles whose lastmod is on or after
    ``since``. If ``topics_filter`` is set, only URLs whose inferred topic is
    in the set are returned. Includes ``general`` topic for modern URLs whose
    path doesn't carry a topic slug.
    """
    candidates: list[tuple[str, date, str]] = []

    for index_url in (SITEMAP_NEWS_INDEX, SITEMAP_ARCHIVE_INDEX):
        try:
            r = session.get(index_url, timeout=30)
            r.raise_for_status()
        except Exception as exc:
            if verbose:
                print(f"  [sitemap index fail] {index_url}: {exc}")
            continue
        index_root = _parse_sitemap(r.content)
        children: list[str] = []
        for sm in index_root.findall("sitemap"):
            loc = sm.findtext("loc")
            lastmod = _parse_iso_date(sm.findtext("lastmod") or "")
            if not loc:
                continue
            # If the entire child's lastmod is older than `since`, skip it.
            # Articles inside might be older still, but lastmod marks newest.
            if lastmod is None or lastmod >= since:
                children.append(loc)

        if verbose:
            print(f"  {index_url.rsplit('/', 1)[1]}: {len(children)} child sitemap(s) in scope")

        for child_url in children:
            try:
                cr = session.get(child_url, timeout=30)
                cr.raise_for_status()
            except Exception as exc:
                if verbose:
                    print(f"  [child sitemap fail] {child_url}: {exc}")
                continue
            child_root = _parse_sitemap(cr.content)
            kept = 0
            for url_el in child_root.findall("url"):
                loc = url_el.findtext("loc")
                if not loc or "/news/" not in loc:
                    continue  # skip /sport, /turkce, /hindi, etc.
                lastmod = _parse_iso_date(url_el.findtext("lastmod") or "")
                if lastmod is None or lastmod < since:
                    continue
                topic = _topic_from_url(loc)
                if topics_filter and topic not in topics_filter:
                    continue
                candidates.append((loc, lastmod, topic))
                kept += 1
            if verbose:
                print(f"    {child_url.rsplit('/', 1)[1]}: {kept} candidate URL(s)")

    return candidates


def scrape_via_sitemap(
    since: date | None = None,
    years: float = 1.0,
    topics: list[str] | None = None,
    sleep: float = 1.0,
    workers: int = 1,
    verbose: bool = True,
) -> BBCStats:
    """Historical backfill of BBC articles via the public sitemap.

    ``since`` (or ``years`` if not given) bounds how far back the discovery
    walks. Coverage is ~2009 onwards if you go all the way (``years=20``).
    Articles already on disk are skipped via the same per-day-file dedupe as
    RSS mode, so backfill + daily RSS coexist without duplicating fetches.
    """
    if since is None:
        since = date.today() - timedelta(days=int(years * 365.25))
    # Topic filtering happens AFTER fetching, since the URL alone usually
    # doesn't carry topic info on modern BBC articles. Pass None to
    # discovery so we fetch every URL and filter by article-page signal.
    discover_filter = None
    keep_topics = set(topics) if topics else None

    stats = BBCStats()
    discovery_session = _new_session()

    if verbose:
        print(f"Discovering BBC sitemap URLs since {since}...")
    candidates = _discover_via_sitemap(discovery_session, since, discover_filter, verbose)
    if verbose:
        print(f"  total candidates: {len(candidates):,}")

    # Sort by lastmod descending so freshest content lands first if the user
    # interrupts a long backfill.
    candidates.sort(key=lambda t: t[1], reverse=True)

    # We don't know the final topic until we fetch the article. Each worker
    # gets a slice of candidates, fetches, decides topic, saves to the right
    # day-file. Day-file caching is keyed on (topic, day) within the worker.
    chunks = _chunk(candidates, max(1, workers))

    def run_chunk(items: list[tuple[str, date, str]]) -> None:
        session = _new_session()
        day_cache: dict[tuple[str, date], dict[str, dict]] = {}
        dirty: set[tuple[str, date]] = set()

        for url, lastmod, _hint_topic in items:
            article_id_raw = _extract_article_id(url)
            if not article_id_raw:
                continue
            try:
                _sleep(sleep)
                ar = session.get(url, timeout=30)
                ar.raise_for_status()
            except Exception as exc:
                if verbose:
                    print(f"  [article fail] {url}: {exc}")
                stats.add(errors=1)
                continue
            # Pass topic="" so _parse_article doesn't pre-set section; we
            # let the page.subsection logic populate it.
            article = _parse_article(url, ar.text, "", fallback_date=lastmod)
            if not article:
                continue

            try:
                actual_day = (
                    date.fromisoformat(article.published_date)
                    if article.published_date else lastmod
                )
            except ValueError:
                actual_day = lastmod

            target_topic = _topic_for_storage(article.section, url)
            if keep_topics and target_topic not in keep_topics:
                continue

            cache_key = (target_topic, actual_day)
            if cache_key not in day_cache:
                day_cache[cache_key] = load_day(SOURCE, target_topic, actual_day) or {}
            if article.id in day_cache[cache_key]:
                stats.add(skipped_already_have=1)
                continue
            day_cache[cache_key][article.id] = asdict(article)
            dirty.add(cache_key)
            stats.add(fetched_articles=1, by_topic={target_topic: 1})

        for (target_topic, day) in dirty:
            save_day(SOURCE, target_topic, day, day_cache[(target_topic, day)])
            stats.add(days_updated=1)

    if workers <= 1 or len(chunks) == 1:
        for chunk in chunks:
            run_chunk(chunk)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_chunk, c) for c in chunks]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    if verbose:
                        print(f"  [worker crashed] {exc}")
                    stats.add(errors=1)

    return stats


def _chunk(items: list, n: int) -> list[list]:
    if n <= 1 or len(items) <= n:
        return [items] if items else []
    out = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return [c for c in out if c]


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
    parser.add_argument(
        "--topics",
        nargs="*",
        help=(
            "Topics to fetch. RSS mode choices: "
            f"{', '.join(TOPICS.keys())}. Sitemap mode also accepts "
            "'general' (URLs whose path doesn't carry a topic slug)."
        ),
    )
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Base seconds between requests (+0-1s jitter).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel topic workers (1 by default; 4-8 for backfill).")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--from-sitemap",
        action="store_true",
        help="Use BBC's XML sitemap for historical backfill (covers ~2009 → today).",
    )
    parser.add_argument(
        "--years", type=float, default=1.0,
        help="With --from-sitemap, how many years back to scrape (default 1).",
    )
    parser.add_argument(
        "--since",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="With --from-sitemap, override --years with an explicit start date.",
    )
    args = parser.parse_args()

    if args.from_sitemap:
        stats = scrape_via_sitemap(
            since=args.since,
            years=args.years,
            topics=args.topics,
            sleep=args.sleep,
            workers=args.workers,
            verbose=not args.quiet,
        )
    else:
        # RSS mode validates topics against the RSS feed list.
        if args.topics:
            invalid = [t for t in args.topics if t not in TOPICS]
            if invalid:
                parser.error(
                    f"Unknown RSS topic(s): {invalid}. "
                    f"Choices: {', '.join(TOPICS.keys())}."
                )
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
