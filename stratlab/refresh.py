"""Refresh the local OHLCV cache.

Usage:
    python -m stratlab.refresh                       # full default universe
    python -m stratlab.refresh --tickers AAPL MSFT   # specific tickers
    python -m stratlab.refresh --start 2015-01-01    # truncate to this start
    python -m stratlab.refresh --interval 1d         # daily (the default)

Cache layout::

    data/market/
      catalog.json
      indices/{sp500,nasdaq100,dow30}.json
      stocks/<gics_sector>/<TICKER>_1d.csv
      etfs/<category>/<TICKER>_1d.csv
      uncategorized/<TICKER>_1d.csv

For each ticker:
- If no cache exists, fetch ``[start, today]`` from yfinance.
- If cache exists but is shallower than ``--start``, re-fetch the full range.
- If cache covers ``--start`` but stops before today, fetch only the gap.
- If cache covers both edges, skip without a network call.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from stratlab.data.catalog import (
    CATALOG_VERSION,
    UNCATEGORIZED,
    build_catalog,
    category_for,
    load_catalog,
    save_catalog,
)
from stratlab.data.provider import (
    CACHE_DIR,
    CATALOG_PATH,
    INDICES_DIR,
    MARKET_DIR,
    _cache_path,
    _invalidate_catalog_cache,
    _merge_cache,
    _normalize,
    _read_cache,
    _write_cache,
)

LEGACY_HOME_CACHE = Path.home() / ".stratlab" / "cache"

# Old cache layout: {symbol}_{interval}_{16hex}.csv. New layout has no hash
# suffix, so anything matching this pattern is an orphan and can be deleted.
_ORPHAN_CACHE_RE = re.compile(r"^.+_\w+_[0-9a-f]{16}\.csv$")

# When --start is not specified, refresh asks yfinance for ``period="max"``,
# which returns each ticker's full available history without forcing a global
# floor (AAPL → 1980, NVDA → 1999, ABNB → 2020 IPO, ^GSPC → 1927). Pass an
# explicit --start to truncate.

# yf.download saturates the connection pool when given 800+ tickers at once
# (DNS resolution failures, dropped responses). Chunk into manageable groups
# so each batch fits well within macOS / urllib defaults. Large date ranges
# compound the issue: 50 tickers × 99 years (~1.2M rows) is the sweet spot
# where Yahoo reliably returns every ticker we asked for.
BATCH_CHUNK_SIZE = 50
INTER_CHUNK_SLEEP = 1.0  # gentle pause between chunks

# Per-ticker retry pace: when the bulk endpoint silently drops a ticker, we
# retry that name alone via yf.Ticker.history(), which is far more reliable.
PER_TICKER_RETRY_SLEEP = 0.3


@dataclass
class RefreshSummary:
    cold: list[str] = field(default_factory=list)        # had no cache
    backfill: list[str] = field(default_factory=list)    # cache existed but didn't cover --start
    warm: list[str] = field(default_factory=list)        # cache existed, extended forward only
    up_to_date: list[str] = field(default_factory=list)  # cache already covered today
    failed: list[str] = field(default_factory=list)      # no data returned
    orphans_removed: int = 0
    migrated_files: int = 0
    new_bars: int = 0
    cache_dir: Path = MARKET_DIR

    def total(self) -> int:
        return (
            len(self.cold)
            + len(self.backfill)
            + len(self.warm)
            + len(self.up_to_date)
            + len(self.failed)
        )


def cleanup_orphan_cache_files(cache_dir: Path = MARKET_DIR) -> int:
    """Delete cache files left behind by the pre-refactor hash-keyed layout.

    The old layout wrote ``{symbol}_{interval}_{hash}.csv`` (one file per
    requested date range); the current layout writes ``{symbol}_{interval}.csv``
    (one file per symbol holding all bars). The current code never reads or
    updates the old files, so they're pure disk waste.

    Walks recursively to also catch orphans inside category subfolders. Also
    sweeps the legacy ``~/.stratlab/cache/`` location if it still exists.
    """
    removed = 0
    for root in {cache_dir, LEGACY_HOME_CACHE}:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if _ORPHAN_CACHE_RE.match(path.name):
                path.unlink()
                removed += 1
    return removed


_NAME_RE = re.compile(r"^(?P<sym>.+?)(?:_(?P<intv>1d|1h|5m|15m|30m|1m|1wk|1mo))?\.csv$")
# Subfolders directly under MARKET_DIR that count as "categorized" — files
# inside them at the right depth are considered already placed.
_TOP_LEVEL_BUCKETS = (
    "stocks", "etfs", "indices", "futures", "forex", "crypto", "uncategorized",
)


def refresh_catalog(verbose: bool = True) -> dict | None:
    """Force-rebuild the catalog, regardless of version. Returns the new dict."""
    return _ensure_catalog(verbose=verbose, force_rebuild=True)


def _ensure_catalog(verbose: bool = True, force_rebuild: bool = False) -> dict | None:
    """Load the catalog, rebuilding when missing or below the current version.

    Version bumps mean we've added asset classes (indices, futures, …) to the
    schema. A stale catalog routes new symbols to ``uncategorized/`` even
    though they have a proper home now, so we proactively rebuild on bumps.
    """
    catalog = None if force_rebuild else load_catalog(CATALOG_PATH)
    needs_rebuild = catalog is None or catalog.get("version", 0) < CATALOG_VERSION
    if not needs_rebuild:
        return catalog
    if verbose:
        action = "Building" if catalog is None else f"Upgrading from v{catalog.get('version', 0)}"
        print(f"{action} catalog to v{CATALOG_VERSION}...")
    try:
        catalog = build_catalog()
        save_catalog(catalog, CATALOG_PATH)
        _invalidate_catalog_cache()
    except Exception as exc:
        if verbose:
            print(f"  Warning: catalog build failed ({exc!r})")
        return catalog
    if verbose:
        n_stocks = len(catalog.get("stocks", {}))
        n_etfs = len(catalog.get("etfs", {}))
        n_indices = len(catalog.get("indices", {}))
        n_futures = len(catalog.get("futures", {}))
        print(
            f"Catalog: {n_stocks} stocks · {n_etfs} ETFs · "
            f"{n_indices} indices · {n_futures} futures"
        )
    return catalog


def _route(symbol: str, original_name: str, catalog: dict | None) -> Path:
    """Where a CSV with this ticker should live in the new layout.

    Adds an ``_1d`` interval suffix if the source filename had none, so we end
    up with consistent ``<TICKER>_1d.csv`` names everywhere.
    """
    category = category_for(symbol, catalog) if catalog else UNCATEGORIZED
    name = original_name
    if not re.search(r"_\w+\.csv$", name):
        name = f"{symbol}_1d.csv"
    return MARKET_DIR / category / name


def migrate_legacy_cache(verbose: bool = True) -> int:
    """Reorganize cache files into the categorized layout.

    Three sources are reconciled:

    1. ``~/.stratlab/cache/*.csv`` — the pre-restructure global cache.
    2. ``MARKET_DIR/<TICKER>.csv`` — naked CSVs at the data root.
    3. ``MARKET_DIR/{etfs,indices,...}/<TICKER>.csv`` — files placed in a
       top-level bucket but not in the right sub-category.

    Files already correctly nested are left alone. Files whose destination
    already exists are skipped (a future refresh will merge if the user wants).
    """
    catalog = _ensure_catalog(verbose=verbose)
    moved = 0
    skipped = 0

    def relocate(src: Path) -> None:
        nonlocal moved, skipped
        m = _NAME_RE.match(src.name)
        if not m:
            return
        symbol = m.group("sym")
        dest = _route(symbol, src.name, catalog)
        if dest.resolve() == src.resolve():
            return  # already in the right place
        if dest.exists():
            # Merge bars from src into dest, then drop src. Handles the
            # legacy case where the same ticker has files in both
            # uncategorized/ and the correct subfolder.
            try:
                existing = _read_cache(dest)
                incoming = _read_cache(src)
                if incoming is None:
                    src.unlink()
                else:
                    merged_df = _merge_cache(existing, incoming)
                    _write_cache(merged_df, dest)
                    src.unlink()
                moved += 1
            except Exception:
                skipped += 1
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    # 1) legacy ~/.stratlab/cache/
    if LEGACY_HOME_CACHE.exists() and MARKET_DIR.resolve() != LEGACY_HOME_CACHE.resolve():
        if verbose:
            legacy_csvs = list(LEGACY_HOME_CACHE.glob("*.csv"))
            if legacy_csvs:
                print(f"Migrating legacy cache: {LEGACY_HOME_CACHE} → {MARKET_DIR}")
        for src in LEGACY_HOME_CACHE.glob("*.csv"):
            relocate(src)
        # JSON index files in the legacy cache: sp500_tickers.json → indices/sp500.json
        INDICES_DIR.mkdir(parents=True, exist_ok=True)
        for src in LEGACY_HOME_CACHE.glob("*.json"):
            dest = INDICES_DIR / src.name.replace("_tickers", "")
            if dest.exists():
                skipped += 1
                continue
            shutil.move(str(src), str(dest))
            moved += 1

    # 2) naked CSVs at the MARKET_DIR root
    if MARKET_DIR.exists():
        for src in MARKET_DIR.glob("*.csv"):
            relocate(src)

    # 3) misplaced files inside a top-level bucket (e.g. data/market/etfs/GLD.csv
    #    rather than data/market/etfs/commodities/GLD.csv)
    for bucket in _TOP_LEVEL_BUCKETS:
        bucket_dir = MARKET_DIR / bucket
        if not bucket_dir.exists():
            continue
        for src in bucket_dir.glob("*.csv"):
            relocate(src)

    if verbose and moved:
        msg = f"  moved {moved} file(s) into categorized layout"
        if skipped:
            msg += f", skipped {skipped} (destination already exists)"
        print(msg)

    return moved


def refresh_universe(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    verbose: bool = True,
) -> RefreshSummary:
    """Bring every ticker in ``tickers`` up to date in the local cache.

    With ``start=None`` (the default), cold fetches use yfinance's
    ``period="max"`` mode, which returns each ticker's complete available
    history (AAPL gets 1980→today, ABNB gets 2020→today, ^GSPC gets
    1927→today). Pass an explicit ``start`` (e.g. ``"2020-01-01"``) to
    truncate. The summary reports which tickers were fetched cold,
    incrementally extended, already up to date, or failed.
    """
    if tickers is None:
        from stratlab import default_universe

        tickers = default_universe()

    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end)
    # Most recent business day on or before end_ts — avoids fruitless weekend
    # fetches when the cache already has Friday's bar.
    last_bday = pd.bdate_range(end=end_ts, periods=1)[0]
    summary = RefreshSummary()

    # One-time housekeeping on every run (idempotent after first):
    # 1. Make sure the catalog is current — rebuild on schema bumps.
    # 2. Migrate any files placed under the wrong path into the right subfolder.
    # 3. Sweep orphan files from the pre-refactor hash-keyed layout.
    _ensure_catalog(verbose=verbose)

    migrated = migrate_legacy_cache(verbose=verbose)
    summary.migrated_files = migrated

    orphans = cleanup_orphan_cache_files(MARKET_DIR)
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
        # Only flag backfill when the user gave an explicit --start the cache
        # doesn't cover. With start=None we trust whatever's already cached
        # (the original cold fetch grabbed period="max" so it's already
        # complete) and only do forward extends.
        needs_backfill = start_ts is not None and cached.index.min() > start_ts
        needs_forward = cached.index.max() < last_bday
        if not needs_backfill and not needs_forward:
            summary.up_to_date.append(sym)
        elif needs_backfill:
            full_fetch.append(sym)
        else:
            forward_only[sym] = cached.index.max() + pd.Timedelta(days=1)

    cold_count = sum(1 for s in full_fetch if cached_by_sym[s] is None)
    backfill_count = len(full_fetch) - cold_count

    fetch_window = "max history" if start is None else f"from {start}"
    if verbose:
        print(f"Cache directory: {CACHE_DIR}")
        print(f"Universe: {len(tickers)} tickers")
        print(
            f"  cold (no cache → fetch {fetch_window})  : {cold_count}\n"
            f"  backfill (cache too short on the left)  : {backfill_count}\n"
            f"  warm (extend forward to {end})          : {len(forward_only)}\n"
            f"  already up to date                      : {len(summary.up_to_date)}"
        )

    if full_fetch:
        if verbose:
            window = "max history" if start is None else f"{start} → {end}"
            print(f"\nFetching {len(full_fetch)} cold/backfill tickers [{window}]...")
        added = _batch_fetch_and_merge(
            full_fetch, start, end, interval, cached_by_sym, summary, verbose=verbose
        )
        summary.new_bars += added
        if verbose:
            print(f"  cold/backfill fetch added {added:,} bars")

    if forward_only:
        earliest = min(forward_only.values()).strftime("%Y-%m-%d")
        if verbose:
            print(f"\nFetching {len(forward_only)} warm tickers [{earliest} → {end}]...")
        added = _batch_fetch_and_merge(
            list(forward_only.keys()), earliest, end, interval, cached_by_sym, summary, verbose=verbose
        )
        summary.new_bars += added
        if verbose:
            print(f"  warm fetch added {added:,} bars")

    if verbose:
        _print_summary(summary)

    return summary


def _batch_fetch_and_merge(
    tickers: list[str],
    start: str | None,
    end: str,
    interval: str,
    cached_by_sym: dict[str, pd.DataFrame | None],
    summary: RefreshSummary,
    verbose: bool = False,
) -> int:
    """Download ``tickers`` in chunks, merging each ticker into its cache.

    Tickers that fail in the first pass are retried once (network blips and
    rate-limit hiccups are common when fetching hundreds of names). Stable
    failures are added to ``summary.failed``.
    """
    if not tickers:
        return 0

    new_bars_total = 0
    failures: list[str] = []

    chunks = [tickers[i:i + BATCH_CHUNK_SIZE] for i in range(0, len(tickers), BATCH_CHUNK_SIZE)]
    for chunk_idx, chunk in enumerate(chunks):
        if verbose and len(chunks) > 1:
            print(f"  chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} tickers)...")
        added, fails = _fetch_chunk(chunk, start, end, interval, cached_by_sym, summary)
        new_bars_total += added
        failures.extend(fails)
        if chunk_idx < len(chunks) - 1:
            time.sleep(INTER_CHUNK_SLEEP)

    # Bulk-batch failures get retried one ticker at a time. The bulk endpoint
    # silently drops random tickers from large responses; per-ticker requests
    # are slower but each call gets its own response and rarely fails.
    if failures:
        if verbose:
            print(
                f"  bulk pass dropped {len(failures)} tickers; retrying "
                "per-ticker (slower, more reliable)..."
            )
        time.sleep(2.0)
        retry_failures: list[str] = []
        for i, sym in enumerate(failures):
            added, ok = _fetch_one_ticker(sym, start, end, interval, cached_by_sym, summary)
            new_bars_total += added
            if not ok:
                retry_failures.append(sym)
            if verbose and (i + 1) % 25 == 0:
                done = i + 1
                print(f"    [{done}/{len(failures)}] recovered "
                      f"{done - len(retry_failures)}, still failing {len(retry_failures)}")
            time.sleep(PER_TICKER_RETRY_SLEEP)
        summary.failed.extend(retry_failures)

    return new_bars_total


_PERIOD_MAX_FALLBACK_START = "2000-01-01"


def _fetch_one_ticker(
    symbol: str,
    start: str | None,
    end: str,
    interval: str,
    cached_by_sym: dict[str, pd.DataFrame | None],
    summary: RefreshSummary,
) -> tuple[int, bool]:
    """Fetch a single ticker via the dedicated history endpoint.

    yf.Ticker.history() is more reliable than yf.download for individual
    names — each call gets its own connection and response, with no
    multi-ticker truncation. Returns ``(bars_added, success)``.

    When ``start`` is None, uses ``period="max"`` for the ticker's full
    available history. Some recently-launched products (Micro Ether
    futures, etc.) reject ``period="max"`` because Yahoo thinks they have
    too little history; in that case we transparently fall back to an
    explicit ``start`` date and yfinance returns whatever exists.
    """
    raw = _try_fetch(symbol, start, end, interval)
    if raw is None or raw.empty:
        if start is None:
            # period="max" wasn't accepted; retry with an explicit start.
            raw = _try_fetch(symbol, _PERIOD_MAX_FALLBACK_START, end, interval)
        if raw is None or raw.empty:
            return 0, False
    try:
        added = _merge_one(symbol, raw, cached_by_sym.get(symbol), interval, summary)
        return added, True
    except Exception:
        return 0, False


def _try_fetch(symbol: str, start: str | None, end: str, interval: str) -> pd.DataFrame | None:
    history_kwargs = dict(interval=interval, auto_adjust=True)
    if start is None:
        history_kwargs["period"] = "max"
    else:
        history_kwargs["start"] = start
        history_kwargs["end"] = end
    try:
        return yf.Ticker(symbol).history(**history_kwargs)
    except Exception:
        return None


def _fetch_chunk(
    tickers: list[str],
    start: str | None,
    end: str,
    interval: str,
    cached_by_sym: dict[str, pd.DataFrame | None],
    summary: RefreshSummary,
) -> tuple[int, list[str]]:
    """Single yf.download for a chunk of tickers. Returns (bars_added, failed).

    When ``start`` is None, asks for ``period="max"`` so each ticker gets its
    full available history rather than truncated to a global floor.
    """
    download_kwargs = dict(
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    if start is None:
        download_kwargs["period"] = "max"
    else:
        download_kwargs["start"] = start
        download_kwargs["end"] = end

    try:
        raw = yf.download(tickers, **download_kwargs)
    except Exception:
        return 0, list(tickers)

    if raw.empty:
        return 0, list(tickers)

    new_bars = 0
    failed: list[str] = []

    if len(tickers) == 1:
        sym = tickers[0]
        try:
            added = _merge_one(sym, raw, cached_by_sym.get(sym), interval, summary)
            new_bars += added
            if added == 0 and (cached_by_sym.get(sym) is None or cached_by_sym[sym].empty):
                failed.append(sym)
        except Exception:
            failed.append(sym)
        return new_bars, failed

    top_level = raw.columns.get_level_values(0) if hasattr(raw.columns, "get_level_values") else []
    for sym in tickers:
        if sym not in top_level:
            failed.append(sym)
            continue
        try:
            added = _merge_one(sym, raw[sym], cached_by_sym.get(sym), interval, summary)
            new_bars += added
        except Exception:
            failed.append(sym)
    return new_bars, failed


def _merge_one(
    symbol: str,
    raw: pd.DataFrame,
    cached: pd.DataFrame | None,
    interval: str,
    summary: RefreshSummary,
) -> int:
    """Merge ``raw`` (one ticker's fresh download) into the on-disk cache.

    Returns the number of new bars added. Raises ``ValueError`` on empty fresh
    data so the caller can treat it as a transient failure (eligible for
    retry); the chunk dispatcher decides whether to mark it as failed.
    """
    fresh = raw.dropna(how="all")
    if fresh.empty:
        raise ValueError(f"no fresh data for {symbol}")

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
    cache_files = list(summary.cache_dir.rglob("*.csv"))
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
    if summary.migrated_files:
        print(f"  files migrated        : {summary.migrated_files}")
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
        default=None,
        help=(
            "Earliest date to fetch for tickers with no cache. Default: "
            "unspecified (yfinance period='max', i.e. each ticker's full "
            "available history)."
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
