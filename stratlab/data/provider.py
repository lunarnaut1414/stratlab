from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".stratlab" / "cache"


def _cache_path(symbol: str, interval: str) -> Path:
    return CACHE_DIR / f"{symbol}_{interval}.csv"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if df.empty:
        return None
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _merge_cache(cached: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    if cached is None or cached.empty:
        return fresh
    merged = pd.concat([cached, fresh])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _covers(cached: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return (
        cached is not None
        and not cached.empty
        and cached.index.min() <= start
        and cached.index.max() >= end
    )


def load_bars(
    symbol: str,
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV bars for a symbol.

    One cache file per (symbol, interval) holds every bar we've fetched. If the
    requested ``[start, end]`` is fully covered by the cache, we slice and
    return without hitting the network. Otherwise we fetch from yfinance,
    merge with anything cached, and re-save.
    """
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    path = _cache_path(symbol, interval)

    cached = _read_cache(path) if use_cache else None
    if _covers(cached, start_ts, end_ts):
        return cached.loc[start_ts:end_ts].copy()

    raw = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=True)
    if raw.empty:
        if cached is not None and not cached.empty:
            return cached.loc[start_ts:end_ts].copy()
        raise ValueError(f"No data returned for {symbol} from {start} to {end}")

    fresh = _normalize(raw)
    merged = _merge_cache(cached, fresh)

    if use_cache:
        _write_cache(merged, path)

    return merged.loc[start_ts:end_ts].copy()
