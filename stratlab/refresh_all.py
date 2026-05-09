"""Run the full daily refresh: market data + all news sources.

Equivalent to running these in sequence::

    python -m stratlab.refresh
    python -m stratlab.news.npr
    python -m stratlab.news.bbc
    python -m stratlab.news.ap

By default the four pipelines run *in parallel* — they hit different
domains (Yahoo, NPR, BBC, AP) so they don't compete on rate-limits.
Wall-clock time becomes ``max(market, npr, bbc, ap)`` instead of their
sum. Output interleaves; pass ``--serial`` for clean ordered output.

Every subroutine is idempotent — only new bars / articles are fetched
on each run. Exits non-zero if any pipeline reported errors.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from stratlab.news import ap as ap_news
from stratlab.news import bbc as bbc_news
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
    args = parser.parse_args()

    today = date.today()
    window_start = today - timedelta(days=7)
    verbose = not args.quiet
    started = time.time()

    if args.serial:
        _print_step_header("STEP 1 / 4: Market data")
        market = _run_market(verbose=verbose)
        _print_step_header("STEP 2 / 4: NPR (date-archive)")
        npr_stats = _run_npr(verbose=verbose, window_start=window_start, today=today)
        npr_news._print_summary(npr_stats, window_start, today)
        _print_step_header("STEP 3 / 4: BBC (RSS)")
        bbc_stats = _run_bbc(verbose=verbose)
        bbc_news._print_summary(bbc_stats)
        _print_step_header("STEP 4 / 4: AP News (topic hubs)")
        ap_stats = _run_ap(verbose=verbose)
        ap_news._print_summary(ap_stats)
    else:
        _print_step_header("Running 4 pipelines in parallel (output will interleave)")
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_market = ex.submit(_run_market, verbose=verbose)
            f_npr = ex.submit(_run_npr, verbose=verbose,
                              window_start=window_start, today=today)
            f_bbc = ex.submit(_run_bbc, verbose=verbose)
            f_ap = ex.submit(_run_ap, verbose=verbose)
            market = f_market.result()
            npr_stats = f_npr.result()
            bbc_stats = f_bbc.result()
            ap_stats = f_ap.result()

        # Print summaries cleanly after all threads have joined.
        npr_news._print_summary(npr_stats, window_start, today)
        bbc_news._print_summary(bbc_stats)
        ap_news._print_summary(ap_stats)

    market_ok = not market.failed
    npr_ok = npr_stats.errors == 0
    bbc_ok = bbc_stats.errors == 0
    ap_ok = ap_stats.errors == 0
    total_articles = (
        npr_stats.fetched_articles + bbc_stats.fetched_articles + ap_stats.fetched_articles
    )
    elapsed = time.time() - started

    print()
    print("=" * 60)
    print(f"Daily refresh complete in {elapsed:.0f}s")
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
