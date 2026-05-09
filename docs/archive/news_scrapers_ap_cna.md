# Archived: AP News & CNA scrapers

Both scrapers were removed from the live `stratlab.news` package
because their **archive depth was insufficient for the project's
needs** (financial sentiment backtesting requires multi-year history).
Their full source is preserved in git history; the patterns are
documented here for future reuse.

## TL;DR — when each one made sense

| Scraper | Lives in | Archive depth | Why originally added | Why removed |
|---|---|---|---|---|
| `news/ap.py` | AP News (apnews.com) | **None** — ~80-100 latest per topic, no archive endpoint | US wire-service breaking news | Replaced functionally by NPR (which *does* have archives back to ~2000) |
| `news/cna.py` | Channel News Asia (channelnewsasia.com) | **None** — ~50 latest only via Google-News submission feed | Asian English perspective | Replaced by Kyodo News English (which has per-year sitemaps back to 2017) |

The general lesson, restated: **before adding a scraper, probe whether
the publisher exposes a true historical archive.** A `sitemap-news-feed`
endpoint sounds like an archive but is usually a Google News submission
feed limited to the last ~24-48 hours of articles. Real archives look
like NPR's date-archive walker, BBC's sitemap-archive index, or Kyodo's
per-year sitemaps.

## Recovery

Full source for both modules sits in git history. To retrieve:

```bash
# Last commit that contained the live modules:
git log --diff-filter=D --name-only --pretty=format:"%H %s" -- stratlab/news/ap.py stratlab/news/cna.py | head -3

# View the files at that commit:
git show <commit>:stratlab/news/ap.py
git show <commit>:stratlab/news/cna.py

# Or restore them in place (then re-wire as needed):
git show <commit>:stratlab/news/ap.py > stratlab/news/ap.py
git show <commit>:stratlab/news/cna.py > stratlab/news/cna.py
```

The implementation patterns below (URLs, parsing, gotchas) are usually
all you need to rebuild rather than restore.

---

## AP News (`apnews.com`)

### Discovery: topic-hub walker

AP doesn't expose a public RSS feed (the legacy `feeds.apnews.com`
domain doesn't resolve, and `?output=rss` returns HTML), and they don't
publish a sitemap covering articles. The only way to enumerate fresh
articles is to fetch each topic's hub page.

```python
TOPICS = {  # storage topic name → AP hub slug
    "business":      "business",
    "world":         "world-news",     # note slug differs from topic name
    "technology":    "technology",
    "politics":      "politics",
    "science":       "science",
    "health":        "health",
    "entertainment": "entertainment",
    "sports":        "sports",
    "us-news":       "us-news",
}

HUB_URL = "https://apnews.com/hub/{slug}"

# Each hub returns ~80-100 article links per page-load — no pagination beyond that.
# Article URLs match: /article/<headline-slug>-<32+ hex chars>
_AP_ID_RE = re.compile(r"/article/[^/]+?-([a-f0-9]{16,})/?(?:\?|$)")
```

To enumerate an article corpus over time, you have to **run this
scraper daily on cron** and accumulate. There's no way to backfill
beyond the latest hub snapshot.

### Article parsing

AP articles are well-formed and have clean meta tags:

```python
def _parse_article(url, html, topic):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.replace(" | AP News", "").strip()

    # Pub date is in the standard OpenGraph article meta:
    pub_date = soup.find("meta", attrs={"property": "article:published_time"})["content"][:10]

    # Authors are de-slugified from URLs of the form /author/<first-last>:
    authors = []
    for meta in soup.find_all("meta", attrs={"property": "article:author"}):
        slug = meta["content"].rstrip("/").split("/")[-1].split("?")[0]
        authors.append(" ".join(w.capitalize() for w in slug.split("-")))

    # Body lives in .RichTextStoryBody (their article-text container class):
    body = soup.select_one(".RichTextStoryBody") or soup.find("main")
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")
                  if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20]

    return Article(
        id=f"{pub_date}-{article_id_raw}",
        url=url, title=title, authors=authors,
        published_date=pub_date, section=topic,
        content=" ".join(paragraphs),
    )
```

### Robots / scrape policy

`apnews.com/robots.txt` blocks named AI training crawlers (GPTBot,
ClaudeBot, anthropic-ai, etc.) but the generic `User-agent: *` allows
articles. Our `stratlab/0.1` UA was treated as generic. No
`Crawl-delay` set; default 1s sleep was sufficient.

### Why we removed it

- **Coverage gap**: only the latest ~100 per topic visible at any
  given moment. To build a 1-year corpus, you'd have to run daily for
  a year. Useless for backfill.
- **Redundant with NPR for English-speaking US news**: NPR offers
  similar wire-style coverage AND has a date-archive walker back to
  ~2000. The marginal value of AP-on-cron over NPR-with-history was
  small.
- Easy to bring back if needed for a "real-time only" product where
  freshness matters more than depth.

---

## Channel News Asia (`channelnewsasia.com`)

### Discovery: sitemap-news-feed (NOT a real archive)

CNA exposes `https://www.channelnewsasia.com/api/v1/sitemap-news-feed`
which returns XML following the
[Google News sitemap](https://developers.google.com/search/docs/crawling-indexing/sitemaps/news-sitemap)
schema. **This is a Google News submission feed, not a historical
archive.** It contains the ~50 most recent articles tagged with
`<news:publication_date>`. Older articles are silently dropped.

```python
SITEMAP_NEWS_FEED = "https://www.channelnewsasia.com/api/v1/sitemap-news-feed"

# CNA's URL path → topic vocabulary. Their topics are in the URL stem:
_PATH_TO_TOPIC = {
    "asia": "asia", "east-asia": "asia",
    "singapore": "singapore", "world": "world",
    "business": "business", "sport": "sport",
    "commentary": "commentary", "lifestyle": "lifestyle",
    "sustainability": "sustainability", "health": "health",
    "wellness": "health",
    "tech": "technology", "technology": "technology",
    "entertainment": "entertainment",
    "cnainsider": "feature", "cna-lifestyle": "lifestyle",
}

def _topic_from_url(url):
    parts = [p for p in urlparse(url).path.split("/") if p]
    return _PATH_TO_TOPIC.get(parts[0] if parts else "", "general")

# Article ID at end of URL path: trailing /<6+digits>/
_ARTICLE_ID_RE = re.compile(r"-(\d{6,})/?$")
```

Fetch + parse pattern:

```python
def _fetch_sitemap_feed(session):
    r = session.get(SITEMAP_NEWS_FEED, timeout=30)
    r.raise_for_status()
    root = _parse_xml(r.content)  # strip namespaces for simpler ET access
    items = []
    for url_el in root.iter("url"):
        loc = url_el.findtext("loc") or ""
        news = url_el.find("news")
        pub_date_str = news.findtext("publication_date") if news is not None else ""
        try:
            pub_date = datetime.fromisoformat(pub_date_str).date() if pub_date_str else None
        except ValueError:
            pub_date = None
        items.append({"url": loc, "pub_date": pub_date,
                      "title": news.findtext("title") if news is not None else "",
                      "keywords": news.findtext("keywords") if news is not None else ""})
    return items
```

### Article parsing

CNA's HTML uses Mediacorp's analytics-tagged author meta and a
`text-long` body container:

```python
# Author from Mediacorp's analytics tag (cXense)
authors = [m["content"].strip()
           for m in soup.find_all("meta", attrs={"name": "cXenseParse:author"})
           if m.get("content")]

# Description from standard og:description
description = soup.find("meta", attrs={"property": "og:description"})["content"]

# Body: paragraph soup inside div.text-long
paragraphs = [p.get_text(" ", strip=True)
              for p in soup.select("div.text-long p")
              if p.get_text(strip=True) and len(p.get_text(strip=True)) > 20]
```

### Robots / scrape policy

CNA's `robots.txt` had `Crawl-delay: 10` for generic crawlers. We
respected it with `--sleep 2` (CNA tolerated this with no 429s in
practice). No AI-bot-specific blocks at the time of original
implementation. Our `stratlab/0.1` UA was treated as generic.

### Why we removed it

- **Hard limit of 50 articles** in the sitemap-news-feed. No deeper
  endpoint discovered.
- **Replaced by Kyodo English** which gives 9 years of archive (2017+,
  per-year sitemaps). Kyodo is a wire service, not a daily news site,
  but the editorial overlap with CNA is substantial — both cover Asia
  with an English audience.
- The original implementation also pivoted away from NHK (which had
  AI-bot blocks at the time) — but that decision was later reversed
  when we re-read NHK's robots.txt and found generic UAs were OK.

### Misconception cleared up post-implementation

The original module docstring said "NHK was considered but their
robots.txt explicitly disallows AI/scraper bots so we go with CNA
instead." This was a misread — NHK only disallows *named* AI training
crawlers (GPTBot, anthropic-ai, etc.), not generic UAs. The same
applies to NPR, BBC, AP, Mainichi, Asahi, and others. The lesson:
**always read robots.txt as `User-agent: *` rules first**, then check
the named-bot blocklists separately. The named-bot rules are not a
ban on all scraping — they're a ban on training-data scraping by
specific commercial AI companies.

---

## Future reuse checklist

If you want to re-add either of these (or a similar daily-only
source), the pattern that survived to production is:

1. **Always have a `_known_article_ids()` index** that scans on-disk
   articles at startup so re-runs skip cached IDs without HTTP. This
   is what makes the BBC and Kyodo backfills resumable.
2. **Always flush periodically** (every ~20 articles or 30 seconds).
   Without this, a Ctrl+C during a multi-hour scrape loses everything
   buffered in memory.
3. **Always include progress + ETA logging** in the form
   `[progress] global X/Y (P%) — fetched, skipped, errors | rate Q/s
   | ETA Hh Mm`. Multi-hour scrapes that go silent feel hung even
   when working.
4. **Daily-only sources should NOT be added to `news/backfill.py`** —
   their `--since` argument can't honor depth their archive doesn't
   expose. Wire them only into `refresh_all.py` for incremental cron
   runs.
5. **Topic taxonomy comes from the publisher, not your wishes.** AP
   uses URL-suffix slugs. CNA uses URL-prefix paths. Kyodo uses
   JSON-LD BreadcrumbList position 2. NPR uses archive-page URL
   parameters. Don't pre-write a `_PATH_TO_TOPIC` map without
   sampling a few real article URLs / pages first.

---

*Archived as of the commit that removed `stratlab/news/ap.py` and
`stratlab/news/cna.py` from the live tree. Use `git log -- stratlab/news/ap.py
stratlab/news/cna.py` to find that commit and recover the full source.*
