from __future__ import annotations

import io
import json
import re
import warnings
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

from stratlab.data._etf_lists import (
    INVERSE_ETFS,
    LEVERAGED_ETFS,
    POPULAR_ETFS,
    SINGLE_STOCK_LEVERAGED_ETFS,
)
from stratlab.data._futures_lists import FUTURES_CATEGORIES
from stratlab.data._index_lists import INDEX_CATEGORIES
from stratlab.data.provider import (
    CACHE_DIR,
    INDICES_DIR,
    _cache_path,
    _covers,
    _merge_cache,
    _normalize,
    _read_cache,
    _write_cache,
)

_USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"

SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)

# US equity tickers: 1-5 uppercase letters, optionally followed by ``-`` and
# 1-2 letters for share classes (BRK-B, BF-B). Filters out SSGA-internal
# identifiers (CUSIP-shaped strings, placeholders) that occasionally appear.
_US_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]{1,2})?$")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NASDAQ100_WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
DOW30_WIKI_URL = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
IWM_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)


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
    cache_path = INDICES_DIR / cache_name
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

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers}, indent=2)
    )
    return tickers


def sp500_tickers(
    use_cache: bool = True,
    max_age_days: int = 7,
    source: str = "spy",
) -> list[str]:
    """Current S&P 500 constituents.

    ``source`` controls where the list comes from:

    - ``"spy"`` (default) — the State Street SPY holdings file. SPY tracks the
      index, so its holdings *are* the basket (one trading day stale). Most
      authoritative free source.
    - ``"wikipedia"`` — scraped from the List_of_S&P_500_companies page.
      Community-maintained, typically updated within hours of S&P
      announcements; reliable but unofficial.

    SPY mode falls back to Wikipedia automatically if State Street's download
    fails (their URL changes occasionally). Tickers are normalized to Yahoo
    format (``BRK.B`` → ``BRK-B``); cash and non-equity holdings are excluded.

    Note: a *current* snapshot — introduces survivorship bias on historical
    backtests, since names that left the index are absent.
    """
    if source == "spy":
        try:
            return _sp500_from_spy(use_cache=use_cache, max_age_days=max_age_days)
        except Exception as exc:
            warnings.warn(
                f"SPY holdings fetch failed ({exc!r}); falling back to Wikipedia.",
                stacklevel=2,
            )
            return _scrape_index_tickers(
                SP500_WIKI_URL, "sp500.json",
                use_cache=use_cache, max_age_days=max_age_days,
            )
    if source == "wikipedia":
        return _scrape_index_tickers(
            SP500_WIKI_URL, "sp500.json",
            use_cache=use_cache, max_age_days=max_age_days,
        )
    raise ValueError(f"unknown source {source!r}; expected 'spy' or 'wikipedia'")


def _sp500_from_spy(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Pull current S&P 500 from the SPY holdings xlsx published by SSGA.

    The file has a few rows of fund metadata above the holdings table, so we
    locate the header row by searching for the ``Ticker`` column rather than
    hardcoding ``skiprows``. Cash, futures, and unidentified rows are dropped.
    """
    cache_path = INDICES_DIR / "sp500.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at < timedelta(days=max_age_days):
            return payload["tickers"]

    resp = requests.get(SPY_HOLDINGS_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()

    # Find the header row by scanning for a cell that says "Ticker".
    raw = pd.read_excel(io.BytesIO(resp.content), header=None, engine="openpyxl")
    header_row = None
    for i, row in raw.iterrows():
        cells = [str(c).strip() for c in row.tolist()]
        if "Ticker" in cells:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not locate 'Ticker' column in SPY holdings file")

    df = pd.read_excel(
        io.BytesIO(resp.content), skiprows=header_row, engine="openpyxl"
    )
    if "Ticker" not in df.columns:
        raise ValueError(
            f"SPY holdings file missing Ticker column; got {df.columns.tolist()}"
        )

    skip_tokens = {"-", "USD", "CASH", "CASH_USD", "NA", "N/A"}
    tickers: list[str] = []
    for raw_t in df["Ticker"].dropna().tolist():
        t = str(raw_t).strip()
        if not t or t.upper() in skip_tokens:
            continue
        # Yahoo uses '-' where SSGA uses '.' (e.g. BRK.B → BRK-B)
        t = t.replace(".", "-")
        if not _US_TICKER_RE.match(t):
            continue  # SSGA-internal identifier, not a tradeable ticker
        tickers.append(t)

    if not tickers:
        raise ValueError("SPY holdings file parsed but yielded no tickers")

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers}, indent=2)
    )
    return tickers


def nasdaq100_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current Nasdaq-100 constituents, scraped from Wikipedia."""
    return _scrape_index_tickers(
        NASDAQ100_WIKI_URL, "nasdaq100.json",
        use_cache=use_cache, max_age_days=max_age_days,
    )


def dow30_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current Dow Jones Industrial Average constituents, scraped from Wikipedia."""
    return _scrape_index_tickers(
        DOW30_WIKI_URL, "dow30.json",
        use_cache=use_cache, max_age_days=max_age_days,
    )


def nasdaq_listed_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Common stocks listed on the Nasdaq exchange — effectively the Nasdaq
    Composite, minus rights/warrants/units/test issues/ETFs.

    Source: ``nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt`` — the official
    daily symbol directory. Filters: Test Issue == N, ETF == N, and ticker
    must match the standard 1-5-letter common-stock pattern (excludes Nasdaq's
    5th-letter classification suffixes for rights/warrants/units/preferreds).
    """
    cache_path = INDICES_DIR / "nasdaq_listed.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at < timedelta(days=max_age_days):
            return payload["tickers"]

    resp = requests.get(NASDAQ_LISTED_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    lines = [l for l in resp.text.splitlines() if not l.startswith("File Creation Time")]
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")
    df = df[df["Test Issue"].astype(str).str.upper() == "N"]
    df = df[df["ETF"].astype(str).str.upper() == "N"]

    # Drop derivative securities sharing the same root ticker as a common stock
    # (rights, warrants, units, preferreds, debt). Nasdaq's 5th-letter scheme
    # is unreliable (GOOGL ends in L), so we filter on Security Name keywords.
    bad_keywords = (
        "Right", "Warrant", "Unit", "Preferred", "Notes",
        "Convertible", "Debenture", "Subordinated",
    )
    def _is_common(name: str) -> bool:
        s = str(name) if name is not None else ""
        return bool(s) and not any(kw in s for kw in bad_keywords)

    df = df[df["Security Name"].apply(_is_common)]

    tickers: list[str] = []
    for raw_t in df["Symbol"].dropna().tolist():
        t = str(raw_t).replace(".", "-").strip()
        if t and _US_TICKER_RE.match(t):
            tickers.append(t)

    if not tickers:
        raise ValueError("nasdaqlisted.txt parsed but yielded no tickers")

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers}, indent=2)
    )
    return tickers


def russell2000_tickers(use_cache: bool = True, max_age_days: int = 7) -> list[str]:
    """Current Russell 2000 constituents, derived from iShares IWM holdings.

    Source: the iShares IWM holdings CSV. IWM is the canonical Russell 2000
    ETF, so its equity holdings are the index. Cash/derivative rows are
    dropped; ticker normalization matches Yahoo (``BRK.B`` → ``BRK-B``).
    """
    cache_path = INDICES_DIR / "russell2000.json"
    if use_cache and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now() - fetched_at < timedelta(days=max_age_days):
            return payload["tickers"]

    resp = requests.get(
        IWM_HOLDINGS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    resp.raise_for_status()

    raw_lines = resp.text.splitlines()
    header_idx = None
    for i, line in enumerate(raw_lines):
        if line.startswith("Ticker,"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate Ticker header in IWM holdings file")

    df = pd.read_csv(io.StringIO("\n".join(raw_lines[header_idx:])))
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]

    tickers: list[str] = []
    for raw_t in df["Ticker"].dropna().tolist():
        t = str(raw_t).replace(".", "-").strip()
        if t and _US_TICKER_RE.match(t):
            tickers.append(t)

    if not tickers:
        raise ValueError("IWM holdings parsed but yielded no tickers")

    INDICES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"fetched_at": datetime.now().isoformat(), "tickers": tickers}, indent=2)
    )
    return tickers


def popular_etfs() -> list[str]:
    """~200 broadly-traded ETFs covering equity, bonds, commodities, REITs,
    currency, factor, thematic, crypto, and volatility. Long-side, unlevered."""
    return list(POPULAR_ETFS)


def inverse_etfs() -> list[str]:
    """~30 inverse / short ETFs (1x, 2x, 3x). Going long these provides short
    exposure to the underlying without margin or borrow."""
    return list(INVERSE_ETFS)


def leveraged_etfs() -> list[str]:
    """~35 leveraged long ETFs (2x, 3x). Daily-rebalanced — multi-day holds
    drift from a simple multiple due to volatility decay."""
    return list(LEVERAGED_ETFS)


def single_stock_leveraged_etfs() -> list[str]:
    """~115 daily-leveraged and inverse single-stock ETFs from Direxion,
    GraniteShares, and REX Shares (T-REX). Each tracks one underlying with
    2x bull / -1x or -2x bear exposure (some 1.25x). Same volatility-decay
    caveat as broad leveraged ETFs."""
    return list(SINGLE_STOCK_LEVERAGED_ETFS)


def _flat(mapping: dict[str, list[str]], *categories: str) -> list[str]:
    seen: dict[str, None] = {}
    cats = categories or tuple(mapping.keys())
    for cat in cats:
        for t in mapping.get(cat, []):
            if t not in seen:
                seen[t] = None
    return list(seen.keys())


def volatility_indices() -> list[str]:
    """VIX, VVIX, MOVE, SKEW, OVX, GVZ, EVZ, plus VIX9D/3M/6M term structure.

    Index *levels* — no ETF-style decay, useful as direct inputs.
    """
    return list(INDEX_CATEGORIES["volatility"])


def equity_indices() -> list[str]:
    """US equity index levels: ^GSPC, ^DJI, ^NDX, ^IXIC, ^RUT, ^MID, ^SML."""
    return list(INDEX_CATEGORIES["equity"])


def international_indices() -> list[str]:
    """International equity index levels: ^FTSE, ^N225, ^HSI, ^GDAXI, etc."""
    return list(INDEX_CATEGORIES["international"])


def rate_indices() -> list[str]:
    """Treasury yield indices: ^IRX (3mo), ^FVX (5y), ^TNX (10y), ^TYX (30y)."""
    return list(INDEX_CATEGORIES["rates"])


def all_indices() -> list[str]:
    """Every index across all categories (vol + equity + intl + rates + currency)."""
    return _flat(INDEX_CATEGORIES)


def commodity_futures() -> list[str]:
    """Continuous contracts for energy, metals, grains, softs, meats, lumber."""
    return _flat(FUTURES_CATEGORIES, "energy", "metals", "grains", "softs", "meats", "lumber")


def equity_index_futures() -> list[str]:
    """E-mini and Micro E-mini equity index futures (ES, NQ, YM, RTY, MES, MNQ)."""
    return list(FUTURES_CATEGORIES["equity_index"])


def rate_futures() -> list[str]:
    """Treasury futures and Fed Funds (ZB, ZN, ZF, ZT, ZQ)."""
    return list(FUTURES_CATEGORIES["rates"])


def currency_futures() -> list[str]:
    """CME currency futures (6E, 6J, 6B, 6S, 6C, 6A, 6N, 6M)."""
    return list(FUTURES_CATEGORIES["currency"])


def all_futures() -> list[str]:
    """Every futures contract across all categories."""
    return _flat(FUTURES_CATEGORIES)


def default_universe(
    include_sp500: bool = True,
    include_nasdaq100: bool = True,
    include_nasdaq_listed: bool = True,
    include_russell2000: bool = True,
    include_dow30: bool = True,
    include_etfs: bool = True,
    include_inverse: bool = True,
    include_leveraged: bool = True,
    include_single_stock_leveraged: bool = True,
    include_indices: bool = True,
    include_futures: bool = True,
) -> list[str]:
    """Combined deduped universe across asset classes.

    Defaults to *everything* — roughly 5000 tickers once Nasdaq-listed and the
    Russell 2000 are in. Toggle flags to scope down (e.g. for a stocks-only
    universe set ``include_futures=False``). Order is preserved so the result
    is reproducible across runs.
    """
    seen: dict[str, None] = {}
    parts: list[list[str]] = []
    if include_sp500:
        parts.append(sp500_tickers())
    if include_nasdaq100:
        parts.append(nasdaq100_tickers())
    if include_nasdaq_listed:
        parts.append(nasdaq_listed_tickers())
    if include_russell2000:
        parts.append(russell2000_tickers())
    if include_dow30:
        parts.append(dow30_tickers())
    if include_etfs:
        parts.append(popular_etfs())
    if include_inverse:
        parts.append(inverse_etfs())
    if include_leveraged:
        parts.append(leveraged_etfs())
    if include_single_stock_leveraged:
        parts.append(single_stock_leveraged_etfs())
    if include_indices:
        parts.append(all_indices())
    if include_futures:
        parts.append(all_futures())

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
