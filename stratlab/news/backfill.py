"""Historical news backfill — separate from the daily refresh.

The daily ``stratlab.refresh_all`` job is intentionally narrow: a ~7-day
incremental window. This module is the opposite — a one-shot job for
pulling deep history from the *only* three sources whose archives
expose it:

- **NPR** has a date-archive walker (``/sections/<topic>/archive?date=...``)
  that goes back to ~2000.
- **BBC** has an XML sitemap covering ~2009-09 → today (~120 child sitemaps).
- **Kyodo News English** has per-year sitemaps covering 2017 → today
  (~10K articles per year).

AP and CNA do not expose historical archives (AP only surfaces ~80-100
recent per topic-hub; CNA's sitemap-news-feed only exposes 50). For
those, run the daily refresh on cron and let it accumulate.

By default the deep sources run sequentially. They hit different
domains, so ``--parallel`` is also safe — left off by default to keep
output legible during multi-day backfills.

Examples::

    # 1 year of all three sources
    python -m stratlab.news.backfill --days 365

    # Absolute date range
    python -m stratlab.news.backfill --since 2020-01-01

    # Only one source
    python -m stratlab.news.backfill --since 2020-01-01 --sources npr
    python -m stratlab.news.backfill --since 2017-01-01 --sources kyodo

    # Crank parallelism within each source
    python -m stratlab.news.backfill --since 2020-01-01 --workers 6
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from stratlab.news import bbc as bbc_news
from stratlab.news import kyodo as kyodo_news
from stratlab.news import npr as npr_news

DEEP_SOURCES = ("npr", "bbc", "kyodo")


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


def _run_kyodo(window_start: date, workers: int, verbose: bool):
    return kyodo_news.scrape(
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
        help=f"Sources to backfill (default: all three — {' '.join(DEEP_SOURCES)}).",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Per-source parallelism — each source distributes work across "
             "its own thread pool internally (default 4).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run all selected sources in parallel rather than sequentially.",
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

    runners = {
        "npr":   lambda: _run_npr(window_start, today, args.workers, verbose),
        "bbc":   lambda: _run_bbc(window_start, args.workers, verbose),
        "kyodo": lambda: _run_kyodo(window_start, args.workers, verbose),
    }
    headers = {
        "npr":   "NPR (date-archive walker)",
        "bbc":   "BBC (XML sitemap)",
        "kyodo": "Kyodo News English (per-year sitemaps)",
    }
    summarizers = {
        "npr":   lambda s: npr_news._print_summary(s, window_start, today),
        "bbc":   bbc_news._print_summary,
        "kyodo": kyodo_news._print_summary,
    }

    selected = [s for s in DEEP_SOURCES if s in args.sources]
    results: dict[str, object] = {}

    if args.parallel and len(selected) > 1:
        _print_step_header(f"Running {' + '.join(selected)} in parallel")
        with ThreadPoolExecutor(max_workers=len(selected)) as ex:
            futures = {ex.submit(runners[s]): s for s in selected}
            for fut, source in list(futures.items()):
                results[source] = fut.result()
        for source in selected:
            summarizers[source](results[source])
    else:
        for source in selected:
            _print_step_header(headers[source])
            results[source] = runners[source]()
            summarizers[source](results[source])

    elapsed = time.time() - started
    all_ok = True
    fetched_total = 0
    print()
    print("=" * 60)
    print(f"Backfill complete in {elapsed/60:.1f} min")
    for source in selected:
        s = results[source]
        ok = s.errors == 0
        all_ok = all_ok and ok
        fetched_total += s.fetched_articles
        print(f"  {source:6s}: {'ok' if ok else 'FAILURES'} "
              f"({s.errors} errors, {s.fetched_articles:,} articles)")
    print(f"  total articles fetched: {fetched_total:,}")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
