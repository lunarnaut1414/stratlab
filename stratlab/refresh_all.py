"""Run the full daily refresh: market data + all news sources.

Equivalent to running these in sequence::

    python -m stratlab.refresh
    python -m stratlab.news.npr
    python -m stratlab.news.bbc
    python -m stratlab.news.ap
    python -m stratlab.news.kyodo

By default the five pipelines run *in parallel* — they hit different
domains (Yahoo, NPR, BBC, AP, Kyodo) so they don't compete on
rate-limits. Wall-clock becomes ``max(market, npr, bbc, ap, kyodo)``
instead of their sum. Output interleaves; pass ``--serial`` for clean
ordered output.

Every subroutine is idempotent — only new bars / articles are fetched
on each run. Exits non-zero if any pipeline reported errors.

CNA was previously included as the Asian source but was deprecated in
favor of Kyodo News English, which has 9 years of public archive vs
CNA's 50-most-recent feed. ``stratlab.news.cna`` still exists and can
be invoked directly if you want, but it's not part of the daily
refresh anymore.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from stratlab.news import ap as ap_news
from stratlab.news import bbc as bbc_news
from stratlab.news import kyodo as kyodo_news
from stratlab.news import npr as npr_news
from stratlab.refresh import refresh_universe


def _run_market(verbose: bool):
    return refresh_universe(verbose=verbose)


def _run_npr(verbose: bool, window_start: date, today: date):
    npr_news.migrate_yearly_to_daily(verbose=verbose)
    return npr_news.scrape(start=window_start, end=today, verbose=verbose)


def _run_bbc(verbose: bool):
    return bbc_news.scrape(verbose=verbose)


def _run_ap(verbose: bool):
    return ap_news.scrape(verbose=verbose)


def _run_kyodo(verbose: bool, window_start: date):
    return kyodo_news.scrape(since=window_start, verbose=verbose)


def _print_step_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run pipelines sequentially (clean ordered output, slower).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-pipeline progress output (just summaries).",
    )
    parser.add_argument(
        "--news-only",
        action="store_true",
        help="Skip the market-data refresh; run all news pipelines in parallel.",
    )
    parser.add_argument(
        "--with-sentiment",
        action="store_true",
        help="After news scrapers finish, score new articles with FinBERT "
             "(requires the [sentiment] extra: torch + transformers).",
    )
    args = parser.parse_args()

    today = date.today()
    window_start = today - timedelta(days=7)
    verbose = not args.quiet
    started = time.time()
    run_market = not args.news_only
    total = 5 if run_market else 4

    market = None
    if args.serial:
        step = 1
        if run_market:
            _print_step_header(f"STEP {step} / {total}: Market data")
            market = _run_market(verbose=verbose)
            step += 1
        _print_step_header(f"STEP {step} / {total}: NPR (date-archive)")
        npr_stats = _run_npr(verbose=verbose, window_start=window_start, today=today)
        npr_news._print_summary(npr_stats, window_start, today)
        step += 1
        _print_step_header(f"STEP {step} / {total}: BBC (RSS)")
        bbc_stats = _run_bbc(verbose=verbose)
        bbc_news._print_summary(bbc_stats)
        step += 1
        _print_step_header(f"STEP {step} / {total}: AP News (topic hubs)")
        ap_stats = _run_ap(verbose=verbose)
        ap_news._print_summary(ap_stats)
        step += 1
        _print_step_header(f"STEP {step} / {total}: Kyodo News English (sitemap)")
        kyodo_stats = _run_kyodo(verbose=verbose, window_start=window_start)
        kyodo_news._print_summary(kyodo_stats)
    else:
        _print_step_header(
            f"Running {total} pipelines in parallel (output will interleave)"
        )
        with ThreadPoolExecutor(max_workers=total) as ex:
            f_market = ex.submit(_run_market, verbose=verbose) if run_market else None
            f_npr = ex.submit(_run_npr, verbose=verbose,
                              window_start=window_start, today=today)
            f_bbc = ex.submit(_run_bbc, verbose=verbose)
            f_ap = ex.submit(_run_ap, verbose=verbose)
            f_kyodo = ex.submit(_run_kyodo, verbose=verbose, window_start=window_start)
            if f_market is not None:
                market = f_market.result()
            npr_stats = f_npr.result()
            bbc_stats = f_bbc.result()
            ap_stats = f_ap.result()
            kyodo_stats = f_kyodo.result()

        # Print summaries cleanly after all threads have joined.
        npr_news._print_summary(npr_stats, window_start, today)
        bbc_news._print_summary(bbc_stats)
        ap_news._print_summary(ap_stats)
        kyodo_news._print_summary(kyodo_stats)

    sentiment_stats = None
    sentiment_ok = True
    if args.with_sentiment:
        from stratlab.news import sentiment as sentiment_mod
        _print_step_header("Sentiment scoring (FinBERT)")
        sentiment_stats = sentiment_mod.score_news_dir(
            since=window_start, verbose=verbose,
        )
        sentiment_mod._print_summary(sentiment_stats)
        sentiment_ok = sentiment_stats.errors == 0

    market_ok = market is None or not market.failed
    npr_ok = npr_stats.errors == 0
    bbc_ok = bbc_stats.errors == 0
    ap_ok = ap_stats.errors == 0
    kyodo_ok = kyodo_stats.errors == 0
    total_articles = (
        npr_stats.fetched_articles + bbc_stats.fetched_articles
        + ap_stats.fetched_articles + kyodo_stats.fetched_articles
    )
    elapsed = time.time() - started

    print()
    print("=" * 60)
    print(f"Daily refresh complete in {elapsed:.0f}s")
    if market is not None:
        print(f"  market: {'ok' if market_ok else 'FAILURES'} "
              f"({len(market.failed)} failed tickers, {market.new_bars:,} new bars)")
    print(f"  npr:    {'ok' if npr_ok else 'FAILURES'} "
          f"({npr_stats.errors} errors, {npr_stats.fetched_articles} articles)")
    print(f"  bbc:    {'ok' if bbc_ok else 'FAILURES'} "
          f"({bbc_stats.errors} errors, {bbc_stats.fetched_articles} articles)")
    print(f"  ap:     {'ok' if ap_ok else 'FAILURES'} "
          f"({ap_stats.errors} errors, {ap_stats.fetched_articles} articles)")
    print(f"  kyodo:  {'ok' if kyodo_ok else 'FAILURES'} "
          f"({kyodo_stats.errors} errors, {kyodo_stats.fetched_articles} articles)")
    print(f"  total news articles fetched: {total_articles}")
    if sentiment_stats is not None:
        print(f"  sentiment: {'ok' if sentiment_ok else 'FAILURES'} "
              f"({sentiment_stats.errors} errors, "
              f"{sentiment_stats.articles_scored} scored)")
    print("=" * 60)

    return 0 if all([market_ok, npr_ok, bbc_ok, ap_ok, kyodo_ok, sentiment_ok]) else 1


if __name__ == "__main__":
    sys.exit(main())
