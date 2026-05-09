"""Run the full daily refresh: market data + news.

Equivalent to running these two in sequence::

    python -m stratlab.refresh
    python -m stratlab.news.npr

Convenient for cron / a daily ritual. Both subroutines are idempotent — only
new bars and new articles are fetched on each run. The script exits non-zero
if either pipeline reported errors so cron can flag failures.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from stratlab.news.npr import _print_summary as _print_news_summary
from stratlab.news.npr import migrate_legacy_npr, scrape
from stratlab.refresh import refresh_universe


def main() -> int:
    today = date.today()
    news_window_start = today - timedelta(days=7)

    print("=" * 60)
    print("STEP 1 / 2: Market data")
    print("=" * 60)
    market = refresh_universe(verbose=True)
    market_ok = not market.failed

    print()
    print("=" * 60)
    print("STEP 2 / 2: News scrape")
    print("=" * 60)
    moved = migrate_legacy_npr()
    if moved:
        print(f"Migrated {moved} legacy NPR file(s) into data/news/npr/")
    news = scrape(start=news_window_start, end=today, verbose=True)
    _print_news_summary(news, news_window_start, today)
    news_ok = news.errors == 0

    print()
    print("=" * 60)
    print("Daily refresh complete")
    print(f"  market: {'ok' if market_ok else 'FAILURES'} "
          f"({len(market.failed)} failed tickers)")
    print(f"  news:   {'ok' if news_ok else 'FAILURES'} "
          f"({news.errors} errors, {news.fetched_articles} articles fetched)")
    print("=" * 60)

    return 0 if (market_ok and news_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
