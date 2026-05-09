from __future__ import annotations

import io
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from stratlab.data._etf_lists import INVERSE_ETFS, LEVERAGED_ETFS, POPULAR_ETFS
from stratlab.data.provider import (
    CACHE_DIR,
    _cache_path,
    _covers,
    _merge_cache,
    _normalize,
    _read_cache,
    _write_cache,
)

_USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
DOW30_WIKI_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"


def _scrape_index_tickers(
    url: str,
    cache_name: str,
    symbol_columns: tuple[str, ...] = ("Symbol", "Ticker", "Ticker symbol"),
    use_cache: bool = True,
    max_age_days: int = 7,
) -> list[str]:
    """Scrape constituent tickers from a Wikipedia index page.

    Picks the first table that has any of ``symbol_columns`` as a column name —
    safer than indexing by table position, since Wikipedia editors rearrange
    tables. Tickers are Yahoo-formatted (``.`` → ``-``).
    """
    cache_path = CACHE_DIR / cache_name
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at < timedelta(days=max_age_days):
            return payload["tickers"]

    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    chosen = None
    chosen_col = None
    for table in tables:
        for col in symbol_columns:
            if col in table.columns:
                chosen = table
                chosen_col = col
                break
        if chosen is not None:
            break

    if chosen is None:
        raise ValueError(
            f"No constituents table found at {url} (looked for columns {symbol_columns})"
        )

    tickers = [str(t).replace(".", "-").strip() for t in chosen[chosen_col].tolist()]
    tickers = [t for t in tickers if t and t.lower() != "nan"]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers})
    )
    return tickers


def sp500_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current S&P 500 constituents, scraped from Wikipedia.

    Tickers are normalized to Yahoo Finance format (e.g. ``BRK.B`` → ``BRK-B``).
    The list is cached to disk and refreshed if older than ``max_age_days``.

    Note: this is a *current* snapshot of the index. Using it on historical
    backtests introduces survivorship bias — the list excludes companies that
    were removed (Lehman, Sears, etc.) and includes companies added recently.
    """
    return _scrape_index_tickers(
        SP500_WIKI_URL, "sp500_tickers.json",
        use_cache=use_cache, max_age_days=max_age_days,
    )


def nasdaq100_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current Nasdaq-100 constituents, scraped from Wikipedia."""
    return _scrape_index_tickers(
        NASDAQ100_WIKI_URL, "nasdaq100_tickers.json",
        use_cache=use_cache, max_age_days=max_age_days,
    )


def dow30_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current Dow Jones Industrial Average constituents, scraped from Wikipedia."""
    return _scrape_index_tickers(
        DOW30_WIKI_URL, "dow30_tickers.json",
        use_cache=use_cache, max_age_days=max_age_days,
    )


def popular_etfs() -> list[str]:
    """~150 broadly-traded ETFs covering equity, bonds, commodities, REITs,
    currency, factor, thematic, crypto, and volatility. Long-side, unlevered."""
    return list(POPULAR_ETFS)


def inverse_etfs() -> list[str]:
    """~30 inverse / short ETFs (1x, 2x, 3x). Going long these provides short
    exposure to the underlying without margin or borrow."""
    return list(INVERSE_ETFS)


def leveraged_etfs() -> list[str]:
    """~25 leveraged long ETFs (2x, 3x). Daily-rebalanced — multi-day holds
    drift from a simple multiple due to volatility decay."""
    return list(LEVERAGED_ETFS)


def default_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    include_dow30: bool = True,
    include_etfs: bool = True,
    include_inverse: bool = True,
    include_leveraged: bool = True,
) -> list[str]:
    """Combined deduped universe of indexes + curated ETF lists.

    Defaults to *everything* — roughly 700 tickers. Toggle the flags to scope
    down. Order is preserved so the result is reproducible across runs.
    """
    seen: dict[str, None] = {}
    parts: list[list[str]] = []
    if include_sp500:
        parts.append(sp500_tickers())
    if include_nasdaq100:
        parts.append(nasdaq100_tickers())
    if include_dow30:
        parts.append(dow30_tickers())
    if include_etfs:
        parts.append(popular_etfs())
    if include_inverse:
        parts.append(inverse_etfs())
    if include_leveraged:
        parts.append(leveraged_etfs())

    for chunk in parts:
        for t in chunk:
            if t and t not in seen:
                seen[t] = None
    return list(seen.keys())


def load_universe(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
    drop_failed: bool = True,
) -> dict[str, pd.DataFrame]:
    """Batch-load OHLCV bars for many tickers.

    Returns ``{ticker: DataFrame}``. Per-ticker frames share the same cache
    layout as :func:`load_bars` — one CSV per (symbol, interval) holding every
    bar we've ever fetched. Cold downloads use ``yfinance``'s threaded batch
    endpoint; cached tickers are sliced without hitting the network.
    """
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    cached_by_sym: dict[str, pd.DataFrame | None] = {}

    for sym in tickers:
        cached = _read_cache(_cache_path(sym, interval)) if use_cache else None
        if _covers(cached, start_ts, end_ts):
            out[sym] = cached.loc[start_ts:end_ts].copy()
        else:
            missing.append(sym)
            cached_by_sym[sym] = cached

    if missing:
        raw = yf.download(
            missing,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if not raw.empty:
            if len(missing) == 1:
                sym = missing[0]
                _absorb(sym, raw, cached_by_sym[sym], interval, use_cache, start_ts, end_ts, out)
            else:
                top_level = raw.columns.get_level_values(0)
                for sym in missing:
                    if sym not in top_level:
                        continue
                    _absorb(
                        sym,
                        raw[sym],
                        cached_by_sym[sym],
                        interval,
                        use_cache,
                        start_ts,
                        end_ts,
                        out,
                    )

    if not drop_failed and len(out) != len(tickers):
        failed = sorted(set(tickers) - set(out.keys()))
        raise ValueError(f"No data returned for: {failed}")

    return out


def _absorb(
    symbol: str,
    raw: pd.DataFrame,
    cached: pd.DataFrame | None,
    interval: str,
    use_cache: bool,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    out: dict[str, pd.DataFrame],
) -> None:
    fresh = raw.dropna(how="all")
    if fresh.empty:
        if cached is not None and not cached.empty:
            out[symbol] = cached.loc[start_ts:end_ts].copy()
        return
    fresh = _normalize(fresh)
    merged = _merge_cache(cached, fresh)
    if use_cache:
        _write_cache(merged, _cache_path(symbol, interval))
    sliced = merged.loc[start_ts:end_ts].copy()
    if not sliced.empty:
        out[symbol] = sliced
