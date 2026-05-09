"""Backfill resume guarantees — verify scrapers don't re-fetch what's on disk.

These lock in the behavior: a deep backfill from years ago should hit
HTTP only for the gaps. Already-scraped days (including empty `{}`
markers) must skip without any network activity.

We don't actually let the scrapers reach the real internet — every test
patches the HTTP session so any request raises loudly. If the resume
logic is broken, the test crashes on the first surprise GET.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from stratlab.news import bbc as bbc_news
from stratlab.news import kyodo as kyodo_news
from stratlab.news import npr as npr_news
from stratlab.news import storage


class _ExplodingSession:
    """A stand-in for requests.Session that fails any GET. If a scraper
    tries to talk to the network, it crashes loudly."""
    def __init__(self):
        self.headers = {}
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(
            f"unexpected HTTP GET — backfill should have skipped this. "
            f"args={args[:1]}"
        )


@pytest.fixture
def isolated_news_dir(tmp_path: Path, monkeypatch):
    """Redirect every news-storage path to a fresh tmp dir for the test."""
    monkeypatch.setattr(storage, "NEWS_DIR", tmp_path)
    monkeypatch.setattr(npr_news, "NPR_DIR", tmp_path / "npr")
    return tmp_path


def _seed_day(news_dir: Path, source: str, topic: str, day: date,
              articles: dict | None = None) -> None:
    """Pre-populate a day-file as if a previous run had scraped it."""
    storage.save_day(source, topic, day, articles or {})


# -----------------------------------------------------------------------------


def test_npr_scrape_zero_http_when_all_days_already_cached(
    isolated_news_dir, monkeypatch
):
    """If every day in the requested window has a file on disk, the NPR
    scraper must make zero HTTP requests."""
    session = _ExplodingSession()
    monkeypatch.setattr(npr_news, "_new_session", lambda: session)

    start = date(2024, 1, 1)
    end = date(2024, 1, 5)
    topics = ["business", "economy"]

    # Pre-populate every (topic, day) the scraper will iterate over.
    for topic in topics:
        d = start
        while d <= end:
            _seed_day(isolated_news_dir, "npr", topic, d,
                     {"art-1": {"title": "stub"}})
            d += timedelta(days=1)

    stats = npr_news.scrape(
        topics=topics, start=start, end=end, sleep=0, workers=1, verbose=False,
    )

    assert session.calls == 0, "scraper hit the network despite cached days"
    assert stats.days_skipped_cached == len(topics) * 5
    assert stats.days_visited == 0
    assert stats.fetched_articles == 0


def test_npr_scrape_skips_empty_day_file_marker(isolated_news_dir, monkeypatch):
    """An empty ``{}`` day file is meaningful — it means 'we visited this
    archive page and it was empty'. The scraper must honor it as 'done'
    just like a populated file."""
    session = _ExplodingSession()
    monkeypatch.setattr(npr_news, "_new_session", lambda: session)

    day = date(2024, 1, 15)
    _seed_day(isolated_news_dir, "npr", "business", day, articles={})

    stats = npr_news.scrape(
        topics=["business"], start=day, end=day, sleep=0, workers=1, verbose=False,
    )
    assert session.calls == 0
    assert stats.days_skipped_cached == 1
    assert stats.days_visited == 0


def test_npr_scrape_only_visits_missing_days(isolated_news_dir, monkeypatch):
    """Mixed state: some days cached, some not. The scraper must skip the
    cached ones and attempt HTTP only for the gaps."""
    visited_urls: list[str] = []

    class _RecordingSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            visited_urls.append(url)
            # Simulate a polite empty response so the scraper saves a `{}` and moves on.
            class _Resp:
                content = b"<html><body></body></html>"
                text = ""
                def raise_for_status(self): pass
            return _Resp()

    monkeypatch.setattr(npr_news, "_new_session", lambda: _RecordingSession())

    start = date(2024, 2, 1)
    end = date(2024, 2, 5)
    # Pre-cache Feb 2 and Feb 4; leave Feb 1, Feb 3, Feb 5 missing.
    for cached_day in (date(2024, 2, 2), date(2024, 2, 4)):
        _seed_day(isolated_news_dir, "npr", "business", cached_day,
                 {"x": {"title": "y"}})

    stats = npr_news.scrape(
        topics=["business"], start=start, end=end, sleep=0, workers=1, verbose=False,
    )

    # Should have made HTTP requests only for the 3 archive pages of missing days
    # (no article-detail follow-ups since the empty HTML returned no links).
    assert len(visited_urls) == 3, \
        f"expected 3 archive fetches, got {len(visited_urls)}: {visited_urls}"
    assert stats.days_skipped_cached == 2
    assert stats.days_visited == 3


def test_npr_scrape_resume_after_partial_backfill(isolated_news_dir, monkeypatch):
    """Simulate a 'crashed mid-backfill' scenario: some days were saved
    before the crash, the rest weren't. A re-run must pick up where
    the prior left off — only fetch the unsaved tail."""
    visited_urls: list[str] = []

    class _RecordingSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            visited_urls.append(url)
            class _Resp:
                content = b"<html></html>"
                text = ""
                def raise_for_status(self): pass
            return _Resp()

    monkeypatch.setattr(npr_news, "_new_session", lambda: _RecordingSession())

    # Backfill window: 10 days. The "first half" already on disk.
    start = date(2024, 3, 1)
    end = date(2024, 3, 10)
    for d_offset in range(5):  # Mar 1-5 cached
        _seed_day(isolated_news_dir, "npr", "world",
                 start + timedelta(days=d_offset),
                 articles={"art": {"title": "t"}})

    stats = npr_news.scrape(
        topics=["world"], start=start, end=end, sleep=0, workers=1, verbose=False,
    )

    assert len(visited_urls) == 5, "should only refetch the 5 uncached tail days"
    assert stats.days_skipped_cached == 5
    assert stats.days_visited == 5


# -----------------------------------------------------------------------------
# BBC sitemap-mode resume tests
# -----------------------------------------------------------------------------

def _seed_bbc_article(news_dir: Path, topic: str, day: date,
                      article_slug: str) -> None:
    """Pre-populate a BBC day-file with one article whose URL slug matches
    ``article_slug``, the part `_extract_article_id` would return."""
    storage.save_day("bbc", topic, day, {
        f"{day.isoformat()}-{article_slug}": {
            "title": "stub", "url": f"https://www.bbc.com/news/articles/{article_slug}",
        },
    })


def test_bbc_known_article_ids_indexes_all_slugs(isolated_news_dir):
    """The slug index used by the deep-backfill skip must enumerate
    every BBC article on disk, regardless of topic or year."""
    _seed_bbc_article(isolated_news_dir, "business", date(2024, 1, 15), "abc12345")
    _seed_bbc_article(isolated_news_dir, "world", date(2009, 9, 4), "xyz67890a")
    _seed_bbc_article(isolated_news_dir, "general", date(2020, 6, 1), "deadbeef")

    known = bbc_news._known_article_ids(news_dir=isolated_news_dir)
    assert known == {"abc12345", "xyz67890a", "deadbeef"}


def test_bbc_sitemap_skips_article_fetch_when_slug_on_disk(
    isolated_news_dir, monkeypatch,
):
    """The big one: deep backfill must not re-fetch articles whose slugs
    are already on disk. With every candidate URL pointing to a known
    slug, the article fetch path must never run."""
    # Candidate URLs the (mocked) sitemap walk surfaces. BBC slugs are
    # 8+ alphanumeric chars so _extract_article_id accepts them.
    candidate_urls = [
        ("https://www.bbc.com/news/articles/abc12345", date(2024, 1, 15), "general"),
        ("https://www.bbc.com/news/articles/xyz67890a", date(2024, 1, 16), "general"),
    ]
    # Pre-populate both slugs on disk.
    _seed_bbc_article(isolated_news_dir, "business", date(2024, 1, 15), "abc12345")
    _seed_bbc_article(isolated_news_dir, "world", date(2024, 1, 16), "xyz67890a")

    article_fetches: list[str] = []

    class _GuardSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            article_fetches.append(url)
            raise AssertionError(
                f"BBC sitemap mode tried to fetch {url} despite the slug "
                f"already being on disk"
            )

    monkeypatch.setattr(bbc_news, "_new_session", lambda: _GuardSession())
    monkeypatch.setattr(
        bbc_news, "_discover_via_sitemap",
        lambda *args, **kwargs: candidate_urls,
    )

    stats = bbc_news.scrape_via_sitemap(
        since=date(2024, 1, 1), sleep=0, workers=1, verbose=False,
    )

    assert article_fetches == [], "scraper hit the network for cached slugs"
    assert stats.skipped_already_have == 2
    assert stats.fetched_articles == 0


def test_bbc_sitemap_only_fetches_unknown_slugs(isolated_news_dir, monkeypatch):
    """Mixed state: candidate URLs cover known + unknown slugs. Only the
    unknowns must trigger an HTTP fetch."""
    candidate_urls = [
        ("https://www.bbc.com/news/articles/known001", date(2024, 1, 15), "general"),
        ("https://www.bbc.com/news/articles/newslug1",   date(2024, 1, 16), "general"),
        ("https://www.bbc.com/news/articles/known002", date(2024, 1, 17), "general"),
        ("https://www.bbc.com/news/articles/newslug2",   date(2024, 1, 18), "general"),
    ]
    _seed_bbc_article(isolated_news_dir, "business", date(2024, 1, 15), "known001")
    _seed_bbc_article(isolated_news_dir, "business", date(2024, 1, 17), "known002")

    fetched_urls: list[str] = []

    class _RecordingSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            fetched_urls.append(url)
            class _Resp:
                # Empty HTML — _parse_article will return None, so the
                # scraper just moves on without trying to save.
                text = "<html><body></body></html>"
                content = b"<html><body></body></html>"
                def raise_for_status(self): pass
            return _Resp()

    monkeypatch.setattr(bbc_news, "_new_session", lambda: _RecordingSession())
    monkeypatch.setattr(
        bbc_news, "_discover_via_sitemap",
        lambda *args, **kwargs: candidate_urls,
    )

    stats = bbc_news.scrape_via_sitemap(
        since=date(2024, 1, 1), sleep=0, workers=1, verbose=False,
    )

    fetched_slugs = sorted(u.rsplit("/", 1)[-1] for u in fetched_urls)
    assert fetched_slugs == ["newslug1", "newslug2"], \
        f"should only fetch unknown slugs, got {fetched_slugs}"
    assert stats.skipped_already_have == 2


# -----------------------------------------------------------------------------
# Kyodo News English sitemap-mode resume tests
# -----------------------------------------------------------------------------

def _seed_kyodo_article(news_dir: Path, topic: str, day: date,
                        article_id: str) -> None:
    """Pre-populate a Kyodo day-file with one article whose URL slug matches
    ``article_id`` (the numeric part after ``/articles/-/``)."""
    storage.save_day("kyodo", topic, day, {
        f"{day.isoformat()}-{article_id}": {
            "title": "stub",
            "url": f"https://english.kyodonews.net/articles/-/{article_id}",
        },
    })


def test_kyodo_known_article_ids_indexes_all_slugs(isolated_news_dir):
    """The Kyodo slug index must enumerate every article ID across topics
    and years."""
    _seed_kyodo_article(isolated_news_dir, "japan", date(2024, 1, 15), "12345")
    _seed_kyodo_article(isolated_news_dir, "world", date(2018, 6, 1), "67")
    _seed_kyodo_article(isolated_news_dir, "business", date(2020, 12, 31), "999999")

    known = kyodo_news._known_article_ids(news_dir=isolated_news_dir)
    assert known == {"12345", "67", "999999"}


def test_kyodo_sitemap_skips_article_fetch_when_id_on_disk(
    isolated_news_dir, monkeypatch,
):
    """Deep backfill must not re-fetch articles whose IDs are already on disk.
    With every candidate's ID pre-cached, the article fetch path must never
    run."""
    candidates = [
        ("https://english.kyodonews.net/articles/-/100001", date(2024, 1, 15)),
        ("https://english.kyodonews.net/articles/-/100002", date(2024, 1, 16)),
    ]
    _seed_kyodo_article(isolated_news_dir, "japan", date(2024, 1, 15), "100001")
    _seed_kyodo_article(isolated_news_dir, "world", date(2024, 1, 16), "100002")

    article_fetches: list[str] = []

    class _GuardSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            article_fetches.append(url)
            raise AssertionError(
                f"Kyodo tried to fetch {url} despite the ID already being on disk"
            )

    monkeypatch.setattr(kyodo_news, "_new_session", lambda: _GuardSession())
    monkeypatch.setattr(
        kyodo_news, "_discover_via_sitemap",
        lambda *args, **kwargs: candidates,
    )

    stats = kyodo_news.scrape(
        since=date(2024, 1, 1), sleep=0, workers=1, verbose=False,
    )

    assert article_fetches == [], "scraper hit the network for cached IDs"
    assert stats.skipped_already_have == 2
    assert stats.fetched_articles == 0


def test_kyodo_sitemap_only_fetches_unknown_ids(isolated_news_dir, monkeypatch):
    """Mixed state: candidates cover known + unknown IDs. Only unknowns
    must trigger HTTP."""
    candidates = [
        ("https://english.kyodonews.net/articles/-/200001", date(2024, 1, 15)),
        ("https://english.kyodonews.net/articles/-/200002", date(2024, 1, 16)),
        ("https://english.kyodonews.net/articles/-/200003", date(2024, 1, 17)),
        ("https://english.kyodonews.net/articles/-/200004", date(2024, 1, 18)),
    ]
    # Pre-cache 200001 and 200003.
    _seed_kyodo_article(isolated_news_dir, "japan", date(2024, 1, 15), "200001")
    _seed_kyodo_article(isolated_news_dir, "japan", date(2024, 1, 17), "200003")

    fetched_urls: list[str] = []

    class _RecordingSession:
        def __init__(self):
            self.headers = {}
        def get(self, url, **kwargs):
            fetched_urls.append(url)
            class _Resp:
                text = "<html><body></body></html>"
                content = b"<html><body></body></html>"
                def raise_for_status(self): pass
            return _Resp()

    monkeypatch.setattr(kyodo_news, "_new_session", lambda: _RecordingSession())
    monkeypatch.setattr(
        kyodo_news, "_discover_via_sitemap",
        lambda *args, **kwargs: candidates,
    )

    stats = kyodo_news.scrape(
        since=date(2024, 1, 1), sleep=0, workers=1, verbose=False,
    )

    fetched_ids = sorted(u.rsplit("/", 1)[-1] for u in fetched_urls)
    assert fetched_ids == ["200002", "200004"], \
        f"should only fetch unknown IDs, got {fetched_ids}"
    assert stats.skipped_already_have == 2
