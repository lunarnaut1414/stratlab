from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf

# --- Cache root resolution -------------------------------------------------

_HOME_CACHE = Path.home() / ".stratlab" / "cache"


def _resolve_cache_root() -> Path:
    """Pick the directory where market data lives.

    Priority:

    1. ``STRATLAB_CACHE_DIR`` env var if set.
    2. ``<project_root>/data/market/`` if the current working directory (or
       any parent) contains a ``pyproject.toml`` or ``.git``. This keeps the
       data lake visible alongside the source code.
    3. ``~/.stratlab/cache/`` as a global fallback.
    """
    env_root = os.environ.get("STRATLAB_CACHE_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").is_dir():
            return parent / "data" / "market"

    return _HOME_CACHE


MARKET_DIR: Path = _resolve_cache_root()
# Legacy alias: existing imports of CACHE_DIR keep working.
CACHE_DIR: Path = MARKET_DIR
INDICES_DIR: Path = MARKET_DIR / "indices"
CATALOG_PATH: Path = MARKET_DIR / "catalog.json"


# --- Cache path resolution -------------------------------------------------

_catalog_singleton: dict | None = None


def _get_catalog() -> dict | None:
    """Lazily load the catalog from disk; rebuild it if missing.

    Cached in-process so we don't re-read the JSON on every cache lookup.
    Returns None only if both the on-disk catalog is missing AND the build
    fails (e.g., offline + no prior catalog) — callers fall back to
    ``uncategorized`` in that case.
    """
    global _catalog_singleton
    if _catalog_singleton is not None:
        return _catalog_singleton

    from stratlab.data.catalog import build_catalog, load_catalog, save_catalog

    catalog = load_catalog(CATALOG_PATH)
    if catalog is None:
        try:
            catalog = build_catalog()
            save_catalog(catalog, CATALOG_PATH)
        except Exception:
            return None

    _catalog_singleton = catalog
    return catalog


def _invalidate_catalog_cache() -> None:
    """Force the next ``_get_catalog()`` call to re-read from disk."""
    global _catalog_singleton
    _catalog_singleton = None


def _cache_path(symbol: str, interval: str) -> Path:
    """Cache file path: ``MARKET_DIR / <category> / <symbol>_<interval>.csv``.

    Category comes from the catalog (``stocks/<gics_sector>`` or
    ``etfs/<category>``). Unknown tickers go to ``uncategorized/``.
    """
    from stratlab.data.catalog import UNCATEGORIZED, category_for

    catalog = _get_catalog()
    category = category_for(symbol, catalog) if catalog else UNCATEGORIZED
    return MARKET_DIR / category / f"{symbol}_{interval}.csv"


# --- Read / write / merge --------------------------------------------------

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # yf.download with group_by="ticker" returns MultiIndex columns like
    # ("AAPL", "Open"). Flatten by keeping only the field-name level so the
    # OHLCV filter below works regardless of single- vs multi-ticker shape.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c).lower() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols]


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Read a cached OHLCV CSV, tolerant of capitalized/legacy column names.

    Files we write use lowercase ``date`` index and OHLCV columns. But we also
    absorb yfinance-raw CSVs (``Date,Open,High,Low,Close,Adj Close,Volume``)
    when migrating user-provided files into the cache, so this reader accepts
    capitalized headers and folds ``Adj Close`` into ``close``.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    # Locate a date-like column (Date / date / Datetime / Timestamp / ...)
    date_col = next((c for c in df.columns if str(c).lower() in ("date", "datetime", "timestamp", "time")), None)
    if date_col is None:
        return None
    df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    if df.empty:
        return None
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"
    df.columns = [str(c).lower() for c in df.columns]
    # Prefer "adj close" (split/dividend adjusted) when both are present.
    if "adj close" in df.columns:
        if "close" in df.columns:
            df = df.drop(columns=["close"])
        df = df.rename(columns={"adj close": "close"})
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not cols:
        return None
    return df[cols]


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


# --- Public API ------------------------------------------------------------

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
