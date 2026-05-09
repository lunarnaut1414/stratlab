"""Run the full daily refresh: market data + all news sources.

Equivalent to running these in sequence::

    python -m stratlab.refresh
    python -m stratlab.news.npr
    python -m stratlab.news.bbc
    python -m stratlab.news.ap

Convenient for cron / a daily ritual. Every subroutine is idempotent — only
new bars / articles are fetched on each run. Exits non-zero if any pipeline
reported errors, so cron can flag failures.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from stratlab.news import ap as ap_news
from stratlab.news import bbc as bbc_news
from stratlab.news import npr as npr_news
from stratlab.refresh import refresh_universe


def _step(num: int, total: int, title: str) -> None:
    print()
    print("=" * 60)
    print(f"STEP {num} / {total}: {title}")
    print("=" * 60)


def main() -> int:
    today = date.today()
    news_window_start = today - timedelta(days=7)
    total_steps = 4

    _step(1, total_steps, "Market data")
    market = refresh_universe(verbose=True)
    market_ok = not market.failed

    _step(2, total_steps, "NPR (date-archive)")
    npr_news.migrate_yearly_to_daily(verbose=True)
    npr_stats = npr_news.scrape(start=news_window_start, end=today, verbose=True)
    npr_news._print_summary(npr_stats, news_window_start, today)
    npr_ok = npr_stats.errors == 0

    _step(3, total_steps, "BBC (RSS)")
    bbc_stats = bbc_news.scrape(verbose=True)
    bbc_news._print_summary(bbc_stats)
    bbc_ok = bbc_stats.errors == 0

    _step(4, total_steps, "AP News (topic hubs)")
    ap_stats = ap_news.scrape(verbose=True)
    ap_news._print_summary(ap_stats)
    ap_ok = ap_stats.errors == 0

    total_articles = (
        npr_stats.fetched_articles + bbc_stats.fetched_articles + ap_stats.fetched_articles
    )

    print()
    print("=" * 60)
    print("Daily refresh complete")
    print(f"  market: {'ok' if market_ok else 'FAILURES'} "
          f"({len(market.failed)} failed tickers, {market.new_bars:,} new bars)")
    print(f"  npr:    {'ok' if npr_ok else 'FAILURES'} "
          f"({npr_stats.errors} errors, {npr_stats.fetched_articles} articles)")
    print(f"  bbc:    {'ok' if bbc_ok else 'FAILURES'} "
          f"({bbc_stats.errors} errors, {bbc_stats.fetched_articles} articles)")
    print(f"  ap:     {'ok' if ap_ok else 'FAILURES'} "
          f"({ap_stats.errors} errors, {ap_stats.fetched_articles} articles)")
    print(f"  total news articles fetched: {total_articles}")
    print("=" * 60)

    return 0 if all([market_ok, npr_ok, bbc_ok, ap_ok]) else 1


if __name__ == "__main__":
    sys.exit(main())
