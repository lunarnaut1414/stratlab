"""Kyodo News English (Japan Wire) scraper.

Kyodo is Japan's largest wire service (Reuters/AP equivalent) and their
English desk publishes ~10K articles/year covering Japan domestic, Asia,
and global news with a strong Japan focus.

Discovery is via Kyodo's per-year sitemap index:

    https://english.kyodonews.net/sitemap.xml      (sitemap index)
        ├── sitemap-static.xml                     (navigation, ignore)
        ├── sitemap-2026.xml                       (~10K URLs/year, lastmod tagged)
        ├── sitemap-2025.xml
        ...
        └── sitemap-2017.xml                       (oldest available year)

Article URLs follow ``/articles/-/<numeric_id>``. The ``/articles/photo/...``
variant of each URL is just the image gallery for the same article and is
explicitly disallowed by ``robots.txt``, so we filter those out.

Articles embed clean JSON-LD with ``headline``, ``datePublished``, ``author``,
and a ``BreadcrumbList`` whose second element gives the topic (e.g. "japan",
"business", "world"). Body lives in ``div.article-body__inner``. Some articles
are paywalled (``isAccessibleForFree: false``) but the lead is still visible.

Same execution pattern as BBC's sitemap mode:

- Pre-flight slug index from on-disk articles → skip already-cached URLs
  before any HTTP fetch
- Periodic flush every 20 articles or 30 seconds → survives Ctrl+C and
  shows visible progress
- Global ETA based on combined fetched + skipped + errors rate
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
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from stratlab.news import storage
from stratlab.news.storage import NEWS_DIR, load_day, save_day

SOURCE = "kyodo"
USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
KYODO_DIR = NEWS_DIR / SOURCE
SITEMAP_INDEX = "https://english.kyodonews.net/sitemap.xml"

# Breadcrumb section name (lowercased) → our storage topic vocabulary.
# Anything not in this map falls back to GENERAL_TOPIC. Add as you see new ones.
GENERAL_TOPIC = "general"
_SECTION_TO_TOPIC: dict[str, str] = {
    "japan": "japan",
    "world": "world",
    "business": "business",
    "sports": "sport",
    "sport": "sport",
    "entertainment": "entertainment",
    "lifestyle": "lifestyle",
    "cool-japan": "culture",
    "culture": "culture",
    "feature": "feature",
    "commentary": "commentary",
    "asia": "asia",
}
TOPICS: tuple[str, ...] = tuple(sorted(set(_SECTION_TO_TOPIC.values()) | {GENERAL_TOPIC}))

_ARTICLE_URL_RE = re.compile(r"/articles/-/(\d+)/?$")
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


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
class KyodoStats:
    sitemaps_walked: int = 0
    candidates_discovered: int = 0
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


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _sleep(base: float, jitter: float = 1.0) -> None:
    if base <= 0:
        return
    time.sleep(base + random.random() * jitter)


def _format_duration(seconds: float) -> str:
    """Compact human duration, e.g. '3d 7h 12m', '4h 9m', '38m', '12s'."""
    if seconds is None or seconds < 0 or seconds != seconds:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h {minutes}m"


def _parse_xml(content: bytes) -> ET.Element:
    """Strip namespaces from XML for simpler ET access."""
    root = ET.fromstring(content)
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _extract_article_id(url: str) -> str | None:
    m = _ARTICLE_URL_RE.search(url)
    return m.group(1) if m else None


def _known_article_ids(news_dir: Path | None = None) -> set[str]:
    """All Kyodo article IDs already on disk, used to short-circuit pre-fetch.

    Day-file keys are ``"<YYYY-MM-DD>-<numeric_id>"`` (or just ``<numeric_id>``
    if the parser couldn't determine a publication date). Strip the date
    prefix to recover the URL-extractable ID.
    """
    if news_dir is None:
        news_dir = storage.NEWS_DIR  # late-bound for test monkeypatching
    root = news_dir / SOURCE
    if not root.exists():
        return set()
    known: set[str] = set()
    for json_file in root.rglob("*.json"):
        try:
            data = _json.loads(json_file.read_text())
        except (OSError, _json.JSONDecodeError):
            continue
        for key in data.keys():
            slug = _DATE_PREFIX_RE.sub("", key, count=1)
            if slug:
                known.add(slug)
    return known


# ---------------------------------------------------------------------------
# Sitemap discovery
# ---------------------------------------------------------------------------

def _discover_via_sitemap(
    session: requests.Session,
    since: date | None,
    verbose: bool,
) -> list[tuple[str, date]]:
    """Walk Kyodo's sitemap index and return ``(url, lastmod)`` candidates.

    ``since`` filters at two levels: skip per-year sitemaps whose latest
    ``lastmod`` is older than ``since``, and within each per-year sitemap
    skip individual URLs older than ``since``. Photo-gallery URLs are
    dropped (robots.txt disallows them and they're duplicates anyway).
    """
    if verbose:
        print(f"Discovering Kyodo sitemap URLs since {since}...")

    r = session.get(SITEMAP_INDEX, timeout=30)
    r.raise_for_status()
    root = _parse_xml(r.content)

    children: list[tuple[str, date | None]] = []
    for sm in root.iter("sitemap"):
        loc = (sm.findtext("loc") or "").strip()
        if not loc or "sitemap-static" in loc:
            continue
        lastmod_str = sm.findtext("lastmod") or ""
        lastmod_d: date | None = None
        if lastmod_str:
            try:
                lastmod_d = datetime.fromisoformat(lastmod_str).date()
            except ValueError:
                pass
        # Per-year sitemap names embed the year, e.g. sitemap-2018.xml. Use
        # that as a coarse "is this year in scope" filter when lastmod is
        # missing or always == today (Kyodo bumps every child's lastmod).
        year_match = re.search(r"sitemap-(\d{4})\.xml$", loc)
        year_int = int(year_match.group(1)) if year_match else None
        if since is not None and year_int is not None:
            # Skip sitemaps for years strictly before since's year.
            if year_int < since.year:
                continue
        children.append((loc, lastmod_d))

    if verbose:
        print(f"  {len(children)} per-year sitemap(s) in scope")

    candidates: list[tuple[str, date]] = []
    for child_url, _child_lastmod in children:
        try:
            cr = session.get(child_url, timeout=60)
            cr.raise_for_status()
            child_root = _parse_xml(cr.content)
        except Exception as exc:
            if verbose:
                print(f"    [child sitemap fail] {child_url}: {exc}")
            continue

        kept = 0
        for url_el in child_root.iter("url"):
            loc = (url_el.findtext("loc") or "").strip()
            if not loc or "/articles/photo/" in loc:
                continue
            if not _ARTICLE_URL_RE.search(loc):
                continue
            lastmod_str = url_el.findtext("lastmod") or ""
            try:
                lastmod = datetime.fromisoformat(lastmod_str).date()
            except ValueError:
                # No usable lastmod — keep the candidate, fall back to
                # 'we'll figure out the date when we parse the article'.
                lastmod = date.today()
            if since is not None and lastmod < since:
                continue
            candidates.append((loc, lastmod))
            kept += 1
        if verbose:
            print(f"    {child_url.rsplit('/', 1)[1]}: {kept:,} candidate URL(s)")

    return candidates


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------

def _extract_jsonld_article(soup: BeautifulSoup) -> dict | None:
    """Return the NewsArticle JSON-LD blob if present."""
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = _json.loads(script.string or "")
        except Exception:
            continue
        candidates = payload if isinstance(payload, list) else [payload]
        for item in candidates:
            if isinstance(item, dict) and item.get("@type") == "NewsArticle":
                return item
    return None


def _topic_from_breadcrumb(soup: BeautifulSoup) -> str:
    """Extract the section topic from the BreadcrumbList JSON-LD.

    The breadcrumb is ``[Top → <section> → article]``; we want element 2.
    Falls back to GENERAL_TOPIC if the structure isn't present.
    """
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = _json.loads(script.string or "")
        except Exception:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict) or item.get("@type") != "BreadcrumbList":
                continue
            elements = item.get("itemListElement") or []
            # The structure can be nested ([[{...}]]) — flatten one level.
            if elements and isinstance(elements[0], list):
                elements = elements[0]
            for el in elements:
                if not isinstance(el, dict):
                    continue
                if el.get("position") == 2:
                    name = (el.get("item") or {}).get("name", "")
                    return _SECTION_TO_TOPIC.get(name.lower().strip(), GENERAL_TOPIC)
    return GENERAL_TOPIC


def _parse_article(url: str, html: str, fallback_date: date | None) -> Article | None:
    soup = BeautifulSoup(html, "html.parser")
    ld = _extract_jsonld_article(soup) or {}

    # Title
    title = (ld.get("headline") or "").strip()
    if not title:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        title = (og_title.get("content") if og_title else "") or ""
        title = title.strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Date — prefer JSON-LD datePublished (ISO 8601 with timezone)
    pub_iso = ld.get("datePublished") or ""
    pub_date = ""
    if pub_iso:
        try:
            pub_date = datetime.fromisoformat(pub_iso).date().isoformat()
        except ValueError:
            pub_date = pub_iso[:10] if len(pub_iso) >= 10 else ""
    if not pub_date and fallback_date is not None:
        pub_date = fallback_date.isoformat()

    # Authors — JSON-LD author[].name; else "KYODO NEWS" wire byline
    authors: list[str] = []
    raw_authors = ld.get("author") or []
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    for a in raw_authors:
        if isinstance(a, dict):
            name = (a.get("name") or "").strip()
            if name and name not in authors:
                authors.append(name)

    # Description
    description = (ld.get("description") or "").strip()
    if not description:
        meta_desc = soup.find("meta", attrs={"property": "og:description"}) or soup.find(
            "meta", attrs={"name": "description"}
        )
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()

    # Body — paragraphs from div.article-body__inner (or article-body fallback)
    body_root = (
        soup.select_one("div.article-body__inner")
        or soup.select_one("div.article-body")
        or soup.find("article")
    )
    paragraphs: list[str] = []
    if body_root:
        for p in body_root.find_all("p"):
            text = p.get_text(" ", strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)
    if not paragraphs:
        # Paywalled or unparseable — but we still have title + description, so
        # synthesize a minimal record from those rather than dropping entirely.
        if not (title and description):
            return None
        paragraphs = [description]

    article_id = _extract_article_id(url)
    if not article_id:
        return None

    topic = _topic_from_breadcrumb(soup)

    return Article(
        id=f"{pub_date}-{article_id}" if pub_date else article_id,
        url=url,
        title=title or url,
        authors=authors,
        published_date=pub_date,
        section=topic,
        content=" ".join(paragraphs),
        description=description,
    )


# ---------------------------------------------------------------------------
# Top-level scrape
# ---------------------------------------------------------------------------

def _chunk(items: list, n: int) -> list[list]:
    if n <= 1 or len(items) <= n:
        return [items] if items else []
    out: list[list] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        out[i % n].append(item)
    return [c for c in out if c]


def scrape(
    since: date | None = None,
    days: int | None = 7,
    topics: list[str] | None = None,
    sleep: float = 1.0,
    workers: int = 4,
    verbose: bool = True,
) -> KyodoStats:
    """Scrape Kyodo English. ``since`` overrides ``days`` if both supplied.

    For daily incremental: leave defaults (``days=7`` covers the recent edge).
    For deep backfill: pass ``since=date(2017, 1, 1)`` (earliest available).
    """
    if since is None and days is not None:
        since = date.today() - timedelta(days=days)

    keep_topics = set(topics) if topics else None
    stats = KyodoStats()
    discovery_session = _new_session()

    candidates = _discover_via_sitemap(discovery_session, since, verbose)
    stats.add(sitemaps_walked=1, candidates_discovered=len(candidates))
    if verbose:
        print(f"  total candidates: {len(candidates):,}")

    # Pre-flight slug index — skip URLs whose article ID is already on disk.
    known_ids = _known_article_ids()
    if verbose and known_ids:
        print(f"  {len(known_ids):,} article IDs already on disk — will skip those URLs")

    if not candidates:
        return stats

    # Sort by lastmod descending so freshest content lands first if interrupted.
    candidates.sort(key=lambda t: t[1], reverse=True)
    chunks = _chunk(candidates, max(1, workers))
    total_candidates = len(candidates)
    backfill_start = time.time()

    FLUSH_EVERY_N_ARTICLES = 20
    FLUSH_EVERY_SECONDS = 30.0

    def run_chunk(items: list[tuple[str, date]]) -> None:
        session = _new_session()
        day_cache: dict[tuple[str, date], dict[str, dict]] = {}
        dirty: set[tuple[str, date]] = set()
        chunk_fetched = 0
        chunk_processed = 0
        last_flush = time.time()

        def flush() -> None:
            for (target_topic, day) in dirty:
                merged = load_day(SOURCE, target_topic, day) or {}
                merged.update(day_cache[(target_topic, day)])
                save_day(SOURCE, target_topic, day, merged)
                day_cache[(target_topic, day)] = merged
                stats.add(days_updated=1)
            dirty.clear()

        for url, lastmod in items:
            chunk_processed += 1
            article_id_raw = _extract_article_id(url)
            if not article_id_raw:
                continue
            if article_id_raw in known_ids:
                stats.add(skipped_already_have=1)
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

            article = _parse_article(url, ar.text, fallback_date=lastmod)
            if not article:
                continue
            if keep_topics and article.section not in keep_topics:
                continue
            try:
                actual_day = (
                    date.fromisoformat(article.published_date)
                    if article.published_date else lastmod
                )
            except ValueError:
                actual_day = lastmod

            cache_key = (article.section, actual_day)
            if cache_key not in day_cache:
                day_cache[cache_key] = load_day(SOURCE, article.section, actual_day) or {}
            if article.id in day_cache[cache_key]:
                stats.add(skipped_already_have=1)
                continue
            day_cache[cache_key][article.id] = asdict(article)
            dirty.add(cache_key)
            stats.add(fetched_articles=1, by_topic={article.section: 1})
            chunk_fetched += 1

            now = time.time()
            if (chunk_fetched > 0
                and (chunk_fetched % FLUSH_EVERY_N_ARTICLES == 0
                     or now - last_flush > FLUSH_EVERY_SECONDS)):
                flush()
                last_flush = now
                if verbose:
                    processed_global = (
                        stats.fetched_articles + stats.skipped_already_have
                        + stats.errors
                    )
                    elapsed = now - backfill_start
                    rate = processed_global / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, total_candidates - processed_global)
                    eta = _format_duration(remaining / rate) if rate > 0 else "?"
                    pct = 100.0 * processed_global / max(total_candidates, 1)
                    print(
                        f"  [progress] global {processed_global:,}/"
                        f"{total_candidates:,} ({pct:.2f}%) — "
                        f"{stats.fetched_articles:,} fetched, "
                        f"{stats.skipped_already_have:,} skipped, "
                        f"{stats.errors:,} errors | "
                        f"rate {rate:.1f}/s | ETA {eta} | flushed"
                    )

        flush()

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


def _print_summary(stats: KyodoStats) -> None:
    print("\n" + "=" * 60)
    print("Kyodo News English scrape summary")
    print(f"  sitemaps walked         : {stats.sitemaps_walked}")
    print(f"  candidates discovered   : {stats.candidates_discovered:,}")
    print(f"  days updated            : {stats.days_updated}")
    print(f"  articles fetched        : {stats.fetched_articles:,}")
    print(f"  skipped (already have)  : {stats.skipped_already_have:,}")
    print(f"  errors                  : {stats.errors}")
    if stats.by_topic:
        print("  by topic                :")
        for topic, n in sorted(stats.by_topic.items(), key=lambda kv: -kv[1]):
            print(f"    {topic:14s}: {n}")
    print(f"  saved to                : {KYODO_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--since", type=str, default=None,
                        help="Absolute backfill start date (YYYY-MM-DD). "
                             "Defaults to today minus --days.")
    parser.add_argument("--days", type=int, default=7,
                        help="Recent days to fetch when --since isn't given (default 7).")
    parser.add_argument("--topics", nargs="*", choices=TOPICS,
                        help=f"Filter by topic (default: all). Choices: {', '.join(TOPICS)}.")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else None
    stats = scrape(
        since=since, days=args.days,
        topics=args.topics, sleep=args.sleep, workers=args.workers,
        verbose=not args.quiet,
    )
    _print_summary(stats)
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
