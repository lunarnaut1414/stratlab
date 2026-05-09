"""Historical news backfill — separate from the daily refresh.

The daily ``stratlab.refresh_all`` job is intentionally narrow: a ~7-day
incremental window. This module is the opposite — a one-shot job for
pulling deep history from the *only* two sources whose archives expose
it:

- **NPR** has a date-archive walker (``/sections/<topic>/archive?date=...``)
  that goes back to ~2000.
- **BBC** has an XML sitemap covering ~2009 → today.

AP and CNA do not expose historical archives, so they're not included
here — for AP/CNA history, run the daily refresh on a cron and let it
accumulate.

By default NPR and BBC run sequentially (serial). They hit different
domains, so ``--parallel`` would also work without rate-limit conflicts;
left off by default to keep output legible during long backfills.

Examples::

    # 1 year of NPR + BBC
    python -m stratlab.news.backfill --days 365

    # Absolute date range
    python -m stratlab.news.backfill --since 2020-01-01

    # Only one source
    python -m stratlab.news.backfill --since 2020-01-01 --sources npr

    # Crank parallelism within each source (per-topic workers)
    python -m stratlab.news.backfill --since 2020-01-01 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from stratlab.news import bbc as bbc_news
from stratlab.news import npr as npr_news

DEEP_SOURCES = ("npr", "bbc")


def _print_step_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _run_npr(window_start: date, today: date, workers: int, verbose: bool):
    npr_news.migrate_yearly_to_daily(verbose=verbose)
    return npr_news.scrape(
        start=window_start, end=today, workers=workers, verbose=verbose,
    )


def _run_bbc(window_start: date, workers: int, verbose: bool):
    return bbc_news.scrape_via_sitemap(
        since=window_start, workers=workers, verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--days", type=int, default=None,
        help="Backfill the last N days. Overridden by --since.",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Absolute start date (YYYY-MM-DD). Required if --days isn't set.",
    )
    parser.add_argument(
        "--sources", nargs="*", choices=DEEP_SOURCES, default=list(DEEP_SOURCES),
        help=f"Sources to backfill (default: {' '.join(DEEP_SOURCES)}).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Per-source parallelism — both NPR and BBC scrape topics in "
             "parallel internally (default 4).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run NPR and BBC in parallel rather than sequentially.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.since and args.days is None:
        parser.error("Specify --days N or --since YYYY-MM-DD.")

    today = date.today()
    if args.since:
        window_start = date.fromisoformat(args.since)
    else:
        window_start = today - timedelta(days=args.days)

    if window_start > today:
        parser.error(f"--since {window_start} is in the future.")

    verbose = not args.quiet
    started = time.time()

    print(f"Backfill window: {window_start} → {today} "
          f"({(today - window_start).days} days)")
    print(f"Sources: {', '.join(args.sources)}  |  workers/source: {args.workers}")

    npr_stats = bbc_stats = None

    def run_npr():
        return _run_npr(window_start, today, args.workers, verbose)

    def run_bbc():
        return _run_bbc(window_start, args.workers, verbose)

    if args.parallel and len(args.sources) > 1:
        _print_step_header("Running NPR + BBC in parallel")
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_npr = ex.submit(run_npr) if "npr" in args.sources else None
            f_bbc = ex.submit(run_bbc) if "bbc" in args.sources else None
            if f_npr is not None:
                npr_stats = f_npr.result()
            if f_bbc is not None:
                bbc_stats = f_bbc.result()
    else:
        if "npr" in args.sources:
            _print_step_header("NPR (date-archive walker)")
            npr_stats = run_npr()
            npr_news._print_summary(npr_stats, window_start, today)
        if "bbc" in args.sources:
            _print_step_header("BBC (XML sitemap)")
            bbc_stats = run_bbc()
            bbc_news._print_summary(bbc_stats)

    if args.parallel:
        if npr_stats is not None:
            npr_news._print_summary(npr_stats, window_start, today)
        if bbc_stats is not None:
            bbc_news._print_summary(bbc_stats)

    elapsed = time.time() - started
    npr_ok = npr_stats is None or npr_stats.errors == 0
    bbc_ok = bbc_stats is None or bbc_stats.errors == 0
    fetched = (
        (npr_stats.fetched_articles if npr_stats else 0)
        + (bbc_stats.fetched_articles if bbc_stats else 0)
    )

    print()
    print("=" * 60)
    print(f"Backfill complete in {elapsed/60:.1f} min")
    if npr_stats is not None:
        print(f"  npr: {'ok' if npr_ok else 'FAILURES'} "
              f"({npr_stats.errors} errors, {npr_stats.fetched_articles} articles)")
    if bbc_stats is not None:
        print(f"  bbc: {'ok' if bbc_ok else 'FAILURES'} "
              f"({bbc_stats.errors} errors, {bbc_stats.fetched_articles} articles)")
    print(f"  total articles fetched: {fetched}")
    print("=" * 60)

    return 0 if (npr_ok and bbc_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
