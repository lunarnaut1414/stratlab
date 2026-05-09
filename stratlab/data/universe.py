from __future__ import annotations

import io
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from stratlab.data.provider import (
    CACHE_DIR,
    _cache_path,
    _covers,
    _merge_cache,
    _normalize,
    _read_cache,
    _write_cache,
)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
TICKERS_CACHE = CACHE_DIR / "sp500_tickers.json"
_USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"


def sp500_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current S&P 500 constituents, scraped from Wikipedia.

    Tickers are normalized to Yahoo Finance format (e.g. ``BRK.B`` → ``BRK-B``).
    The list is cached to disk and refreshed if older than ``max_age_days``.

    Note: this is a *current* snapshot of the index. Using it on historical
    backtests introduces survivorship bias — the list excludes companies that
    were removed (Lehman, Sears, etc.) and includes companies added recently.
    """
    if use_cache and TICKERS_CACHE.exists():
        payload = json.loads(TICKERS_CACHE.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at < timedelta(days=max_age_days):
            return payload["tickers"]

    resp = requests.get(SP500_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    constituents = tables[0]
    tickers = [str(t).replace(".", "-").strip() for t in constituents["Symbol"].tolist()]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TICKERS_CACHE.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers})
    )
    return tickers


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
