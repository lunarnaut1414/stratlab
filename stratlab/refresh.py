"""Refresh the local OHLCV cache.

Usage:
    python -m stratlab.refresh                       # full default universe
    python -m stratlab.refresh --tickers AAPL MSFT   # specific tickers
    python -m stratlab.refresh --start 2015-01-01    # backfill more history
    python -m stratlab.refresh --interval 1d         # daily (the default)

For each ticker:
- If no cache exists, fetch ``[start, today]`` from yfinance.
- If cache exists, fetch only ``[last_cached_date + 1, today]`` and append.

All fetches go through ``yf.download``'s threaded batch endpoint, grouped by
"needs full" vs. "needs partial" so cold tickers and warm tickers don't share
a download. Cache files live in ``~/.stratlab/cache/`` as one CSV per
``(symbol, interval)``.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yfinance as yf

from stratlab.data.provider import (
    CACHE_DIR,
    _cache_path,
    _merge_cache,
    _normalize,
    _read_cache,
    _write_cache,
)

# Old cache layout: {symbol}_{interval}_{16hex}.csv. New layout has no hash
# suffix, so anything matching this pattern is an orphan and can be deleted.
_ORPHAN_CACHE_RE = re.compile(r"^.+_\w+_[0-9a-f]{16}\.csv$")

# Sentinel for "as far back as Yahoo will give us." yfinance accepts arbitrary
# start dates and returns whatever history exists per ticker.
MAX_HISTORY_START = "1900-01-01"


@dataclass
class RefreshSummary:
    cold: list[str] = field(default_factory=list)        # had no cache
    backfill: list[str] = field(default_factory=list)    # cache existed but didn't cover --start
    warm: list[str] = field(default_factory=list)        # cache existed, extended forward only
    up_to_date: list[str] = field(default_factory=list)  # cache already covered today
    failed: list[str] = field(default_factory=list)      # no data returned
    orphans_removed: int = 0
    new_bars: int = 0
    cache_dir: Path = CACHE_DIR

    def total(self) -> int:
        return (
            len(self.cold)
            + len(self.backfill)
            + len(self.warm)
            + len(self.up_to_date)
            + len(self.failed)
        )


def cleanup_orphan_cache_files(cache_dir: Path = CACHE_DIR) -> int:
    """Delete cache files left behind by the pre-refactor hash-keyed layout.

    The old layout wrote ``{symbol}_{interval}_{hash}.csv`` (one file per
    requested date range); the current layout writes ``{symbol}_{interval}.csv``
    (one file per symbol holding all bars). The current code never reads or
    updates the old files, so they're pure disk waste.

    Returns the count of files removed.
    """
    if not cache_dir.exists():
        return 0
    removed = 0
    for path in cache_dir.glob("*.csv"):
        if _ORPHAN_CACHE_RE.match(path.name):
            path.unlink()
            removed += 1
    return removed


def refresh_universe(
    tickers: list[str] | None = None,
    start: str = MAX_HISTORY_START,
    end: str | None = None,
    interval: str = "1d",
    verbose: bool = True,
) -> RefreshSummary:
    """Bring every ticker in ``tickers`` up to date in the local cache.

    Defaults to ``start=1900-01-01`` so each ticker gets its full Yahoo history
    on a cold fetch (AAPL gets 1980→today, ABNB gets 2020→today, etc.). If
    ``tickers`` is None, refreshes :func:`stratlab.default_universe`. The
    summary returned reports which tickers were fetched cold, which were
    incrementally extended, which were already up to date, and which failed.
    """
    if tickers is None:
        from stratlab import default_universe

        tickers = default_universe()

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # Most recent business day on or before end_ts — avoids fruitless weekend
    # fetches when the cache already has Friday's bar.
    last_bday = pd.bdate_range(end=end_ts, periods=1)[0]
    summary = RefreshSummary()

    # Always sweep orphan files from the pre-refactor cache layout. They're
    # provably never read or updated by current code, so this is safe.
    orphans = cleanup_orphan_cache_files(CACHE_DIR)
    summary.orphans_removed = orphans
    if verbose and orphans:
        print(f"Removed {orphans} orphan cache files from old hash-keyed layout")

    # Categorize. A ticker is "up to date" only if its cache covers BOTH the
    # requested start and the latest business day. If either edge is short, we
    # fetch — backfill goes through the full-range path so a single yfinance
    # call covers the gap; pure forward-extends use a tighter range.
    full_fetch: list[str] = []                 # full [start, today] range
    forward_only: dict[str, pd.Timestamp] = {} # symbol → cache_end + 1day
    cached_by_sym: dict[str, pd.DataFrame | None] = {}

    for sym in tickers:
        cached = _read_cache(_cache_path(sym, interval))
        cached_by_sym[sym] = cached
        if cached is None or cached.empty:
            full_fetch.append(sym)
            continue
        needs_backfill = cached.index.min() > start_ts
        needs_forward = cached.index.max() < last_bday
        if not needs_backfill and not needs_forward:
            summary.up_to_date.append(sym)
        elif needs_backfill:
            full_fetch.append(sym)
        else:
            forward_only[sym] = cached.index.max() + pd.Timedelta(days=1)

    cold_count = sum(1 for s in full_fetch if cached_by_sym[s] is None)
    backfill_count = len(full_fetch) - cold_count

    if verbose:
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Universe: {len(tickers)} tickers")
        print(
            f"  cold (no cache → fetch from {start})    : {cold_count}\n"
            f"  backfill (cache too short on the left)  : {backfill_count}\n"
            f"  warm (extend forward to {end})          : {len(forward_only)}\n"
            f"  already up to date                      : {len(summary.up_to_date)}"
        )

    if full_fetch:
        if verbose:
            print(f"\nFetching {len(full_fetch)} cold/backfill tickers [{start} → {end}]...")
        added = _batch_fetch_and_merge(
            full_fetch, start, end, interval, cached_by_sym, summary
        )
        summary.new_bars += added
        if verbose:
            print(f"  cold/backfill fetch added {added:,} bars")

    if forward_only:
        earliest = min(forward_only.values()).strftime("%Y-%m-%d")
        if verbose:
            print(f"\nFetching {len(forward_only)} warm tickers [{earliest} → {end}]...")
        added = _batch_fetch_and_merge(
            list(forward_only.keys()), earliest, end, interval, cached_by_sym, summary
        )
        summary.new_bars += added
        if verbose:
            print(f"  warm fetch added {added:,} bars")

    if verbose:
        _print_summary(summary)

    return summary


def _batch_fetch_and_merge(
    tickers: list[str],
    start: str,
    end: str,
    interval: str,
    cached_by_sym: dict[str, pd.DataFrame | None],
    summary: RefreshSummary,
) -> int:
    if not tickers:
        return 0

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    new_bars_total = 0
    if raw.empty:
        summary.failed.extend(tickers)
        return 0

    if len(tickers) == 1:
        sym = tickers[0]
        added = _merge_one(sym, raw, cached_by_sym.get(sym), interval, summary)
        new_bars_total += added
    else:
        top_level = raw.columns.get_level_values(0)
        for sym in tickers:
            if sym not in top_level:
                summary.failed.append(sym)
                continue
            added = _merge_one(sym, raw[sym], cached_by_sym.get(sym), interval, summary)
            new_bars_total += added

    return new_bars_total


def _merge_one(
    symbol: str,
    raw: pd.DataFrame,
    cached: pd.DataFrame | None,
    interval: str,
    summary: RefreshSummary,
) -> int:
    fresh = raw.dropna(how="all")
    if fresh.empty:
        summary.failed.append(symbol)
        return 0

    fresh = _normalize(fresh)
    new_count = len(fresh) if cached is None else len(fresh.index.difference(cached.index))
    merged = _merge_cache(cached, fresh)
    _write_cache(merged, _cache_path(symbol, interval))

    if cached is None or cached.empty:
        summary.cold.append(symbol)
    elif not fresh.index.empty and fresh.index.min() < cached.index.min():
        summary.backfill.append(symbol)
    else:
        summary.warm.append(symbol)
    return new_count


def _print_summary(summary: RefreshSummary) -> None:
    cache_files = list(summary.cache_dir.glob("*.csv"))
    total_size_mb = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f"Refreshed {summary.total()} tickers")
    print(f"  cold (initial fetch)  : {len(summary.cold)}")
    print(f"  backfill (history)    : {len(summary.backfill)}")
    print(f"  warm (extended cache) : {len(summary.warm)}")
    print(f"  already up to date    : {len(summary.up_to_date)}")
    print(f"  failed                : {len(summary.failed)}")
    if summary.failed:
        preview = summary.failed[:10]
        more = "" if len(summary.failed) <= 10 else f" ... +{len(summary.failed) - 10} more"
        print(f"    failed tickers      : {preview}{more}")
    if summary.orphans_removed:
        print(f"  orphan files removed  : {summary.orphans_removed}")
    print(f"\nNew bars added         : {summary.new_bars:,}")
    print(f"Cache files             : {len(cache_files)}")
    print(f"Cache size              : {total_size_mb:.1f} MB")
    print(f"Location                : {summary.cache_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="Explicit tickers to refresh. Default: full default_universe().",
    )
    parser.add_argument(
        "--start",
        default=MAX_HISTORY_START,
        help=(
            "Earliest date to fetch for tickers with no cache (default: "
            f"{MAX_HISTORY_START}, i.e. max available history per ticker)."
        ),
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (default: today).",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Bar interval (default: 1d).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output; only the final summary is printed.",
    )
    args = parser.parse_args()

    summary = refresh_universe(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        verbose=not args.quiet,
    )
    return 0 if not summary.failed else 1


if __name__ == "__main__":
    sys.exit(main())
