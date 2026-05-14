"""Per-ticker inception dates derived from cached OHLCV.

For each cached symbol, the earliest bar date is effectively its IPO (or as
far back as yfinance has it). Useful for two things:

1. **Pre-filtering a universe** before running a backtest, so the engine
   doesn't carry NaN-tradeable rows for stocks that didn't exist yet
   (UBER pre-2019, ABNB pre-2020, etc.). Eliminates the noisy
   "Failed to get ticker" yfinance warnings during cold downloads.
2. **Inventory** — running ``python -m stratlab.data.inception`` prints
   ``ticker, start_date, end_date, n_bars`` for every cached symbol so
   you can see what the local data lake actually covers.

**Limit**: this only mitigates ONE source of survivorship bias — newly
listed tickers being fictitiously "available" before their IPO. It does
NOT bring back tickers that left the index (Lehman, Bear Stearns, GE
pre-2018, etc.) — those aren't in the cache at all. For full
point-in-time correctness you'd need historical S&P 500 membership data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from stratlab.data.provider import _cache_path


def _read_first_row_date(path: Path) -> pd.Timestamp | None:
    """Cheaply read the first data row's date from a cached CSV.

    Reads at most 2 rows (header + first data) so this is fast across
    hundreds of files.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, nrows=1)
    except Exception:
        return None
    if df.empty:
        return None
    date_col = next(
        (c for c in df.columns if str(c).lower() in ("date", "datetime", "timestamp", "time")),
        None,
    )
    if date_col is None:
        return None
    try:
        ts = pd.to_datetime(df[date_col].iloc[0], errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _read_cache_summary(path: Path) -> dict | None:
    """Return ``{start, end, n_bars}`` for a cached file, or None."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    date_col = next(
        (c for c in df.columns if str(c).lower() in ("date", "datetime", "timestamp", "time")),
        None,
    )
    if date_col is None:
        return None
    idx = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if idx.empty:
        return None
    return {
        "start": idx.min().date(),
        "end": idx.max().date(),
        "n_bars": int(len(idx)),
    }


def ticker_start_dates(
    symbols: list[str],
    interval: str = "1d",
) -> dict[str, date]:
    """For each cached ticker, return its earliest cached bar date.

    Tickers without a cache entry are silently omitted from the result —
    callers should treat ``key in result`` as the existence check.
    """
    out: dict[str, date] = {}
    for sym in symbols:
        ts = _read_first_row_date(_cache_path(sym, interval))
        if ts is not None:
            out[sym] = ts.date()
    return out


def coverage_summary(
    symbols: list[str],
    interval: str = "1d",
) -> dict[str, dict]:
    """Return ``{symbol: {start, end, n_bars}}`` for every cached ticker."""
    out: dict[str, dict] = {}
    for sym in symbols:
        s = _read_cache_summary(_cache_path(sym, interval))
        if s is not None:
            out[sym] = s
    return out


def filter_universe_by_inception(
    tickers: list[str],
    as_of: date | str,
    *,
    min_history_days: int = 0,
    interval: str = "1d",
) -> list[str]:
    """Return only tickers whose cache contains a bar on or before
    ``as_of - min_history_days`` days.

    Use ``min_history_days=252`` for strategies with up-to-1-year lookbacks
    so the warmup is satisfied at IS start. Default 0 just requires the
    stock existed by ``as_of``.

    Tickers not in the cache are excluded.

    **Note**: this is the strict "exists at start of window" filter. For most
    backtests you want :func:`filter_universe_by_window_overlap` instead —
    that one correctly handles mid-window IPOs (UBER at OOS-start, etc.) by
    keeping any ticker whose cache overlaps the window at all.
    """
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of)
    threshold = pd.Timestamp(as_of) - pd.Timedelta(days=min_history_days)
    starts = ticker_start_dates(tickers, interval=interval)
    return [t for t in tickers if t in starts and pd.Timestamp(starts[t]) <= threshold]


def filter_universe_by_window_overlap(
    tickers: list[str],
    start: date | str,
    end: date | str,
    *,
    interval: str = "1d",
) -> list[str]:
    """Return tickers whose cached data overlaps ``[start, end]`` at all.

    This is the right filter for backtests: it excludes tickers with NO data
    in the window (UBER for a 2010-2018 IS run) AND keeps tickers that IPO'd
    mid-window (UBER for a 2019-2026 OOS run). The engine's per-bar NaN
    filtering then handles the pre-IPO portion correctly — the stock is just
    untradeable until its inception date.

    Tickers not in the cache are excluded.
    """
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    summaries = coverage_summary(tickers, interval=interval)
    keep: list[str] = []
    for t in tickers:
        s = summaries.get(t)
        if s is None:
            continue
        cache_start = pd.Timestamp(s["start"])
        cache_end = pd.Timestamp(s["end"])
        # Overlap test: cache_start <= window_end AND cache_end >= window_start
        if cache_start <= end_ts and cache_end >= start_ts:
            keep.append(t)
    return keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--universe", default="sp500",
        help='Universe to inventory: "sp500" (default), "popular_etfs", or "all". '
             'Ignored if --tickers is given.',
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Inspect a specific list of tickers (e.g. --tickers MTUM QUAL VIG). "
             "Overrides --universe. Useful for factor-ETF cache-coverage checks "
             "before designing strategies around them.",
    )
    parser.add_argument(
        "--interval", default="1d",
        help="Bar interval (default 1d).",
    )
    parser.add_argument(
        "--as-of", default=None,
        help="Show only tickers cached as of this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-history-days", type=int, default=0,
        help="Combined with --as-of, require this many days of history before it.",
    )
    parser.add_argument(
        "--missing", action="store_true",
        help="Print only the tickers that have NO cached data.",
    )
    parser.add_argument(
        "--covers-is", action="store_true",
        help="For each ticker, print whether its cached start date is on or "
             "before the IS window start (config.IS_START). Useful for "
             "factor-ETF coverage gap checks (MTUM, QUAL, VIG, etc.).",
    )
    args = parser.parse_args(argv)

    from stratlab.data.universe import (
        sp500_tickers, popular_etfs, default_universe,
    )

    if args.tickers:
        tickers = list(args.tickers)
    elif args.universe == "sp500":
        tickers = sp500_tickers()
    elif args.universe == "popular_etfs":
        tickers = popular_etfs()
    elif args.universe == "all":
        tickers = default_universe()
    else:
        sys.stderr.write(f"unknown universe: {args.universe}\n")
        return 2

    summaries = coverage_summary(tickers, interval=args.interval)

    if args.missing:
        missing = [t for t in tickers if t not in summaries]
        for t in missing:
            print(t)
        sys.stderr.write(f"\n[inception] {len(missing)}/{len(tickers)} tickers missing from cache\n")
        return 0

    if args.as_of:
        eligible = set(filter_universe_by_inception(
            tickers, args.as_of, min_history_days=args.min_history_days,
            interval=args.interval,
        ))
    else:
        eligible = set(summaries.keys())

    is_start = None
    if args.covers_is:
        from stratlab.arena import config as arena_config
        is_start = arena_config.IS_START

    rows = []
    for t in tickers:
        s = summaries.get(t)
        if s is None:
            continue
        row = {
            "ticker": t,
            "start": s["start"].isoformat(),
            "end": s["end"].isoformat(),
            "n_bars": s["n_bars"],
            "eligible": t in eligible,
        }
        if is_start is not None:
            row["covers_is"] = s["start"] <= is_start
        rows.append(row)

    if not rows:
        sys.stderr.write("[inception] no cached tickers among requested\n")
    df = pd.DataFrame(rows).sort_values("start")
    print(df.to_string(index=False))

    sys.stderr.write(
        f"\n[inception] {len(summaries)}/{len(tickers)} cached, "
        f"{sum(1 for r in rows if r['eligible'])} eligible "
        f"as of {args.as_of or 'now'} (min_history_days={args.min_history_days})\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
