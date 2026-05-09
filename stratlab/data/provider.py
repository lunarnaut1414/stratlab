from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path.home() / ".stratlab" / "cache"


def _cache_key(symbol: str, start: str, end: str, interval: str) -> Path:
    h = hashlib.sha256(f"{symbol}:{start}:{end}:{interval}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{symbol}_{interval}_{h}.csv"


def load_bars(
    symbol: str,
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV bars for a symbol. Caches to disk as parquet."""
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    cache_path = _cache_key(symbol, start, end, interval)

    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col="date", parse_dates=True)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} from {start} to {end}")

    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"

    # keep only OHLCV columns
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

    return df
