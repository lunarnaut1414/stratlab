"""Curated lists of index time series, by category.

These are the underlying *index levels* (Yahoo ``^`` prefix) — volatility
indices, treasury yields, and equity index levels. Distinct from the JSON
ticker-list files in ``indices/lists/``: those say "what's IN the index", these
are the index levels themselves.

Volatility indices in particular are useful as inputs because they have no
ETF-style decay (unlike VXX or UVXY) and behave as a clean reference series.
"""
from __future__ import annotations


INDEX_CATEGORIES: dict[str, list[str]] = {
    "volatility": [
        "^VIX",       # CBOE 30-day implied vol on S&P 500
        "^VVIX",      # vol of VIX (vol of vol)
        "^VIX9D",     # 9-day VIX (short term)
        "^VIX3M",     # 3-month VIX
        "^VIX6M",     # 6-month VIX
        "^SKEW",      # CBOE Skew Index — tail risk
        "^OVX",       # CBOE Crude Oil VIX
        "^GVZ",       # CBOE Gold VIX
        "^EVZ",       # CBOE Euro Currency VIX
        "^MOVE",      # ICE BofA Treasury volatility
    ],
    "equity": [
        "^GSPC",      # S&P 500 (canonical Yahoo symbol; ^SPX is an alias)
        "^DJI",       # Dow Jones Industrial Average
        "^IXIC",      # NASDAQ Composite
        "^NDX",       # NASDAQ-100
        "^RUT",       # Russell 2000
        "^MID",       # S&P 400 Mid Cap
        "^SML",       # S&P 600 Small Cap
        "^NYA",       # NYSE Composite
        "^XAX",       # NYSE AMEX Composite
    ],
    "international": [
        "^FTSE",      # FTSE 100 (UK)
        "^N225",      # Nikkei 225 (Japan)
        "^HSI",       # Hang Seng (Hong Kong)
        "^GDAXI",     # DAX (Germany)
        "^FCHI",      # CAC 40 (France)
        "^STOXX50E",  # Euro Stoxx 50
        "^STOXX",     # Euro Stoxx 600
        "^AXJO",      # ASX 200 (Australia)
        "^GSPTSE",    # S&P/TSX Composite (Canada)
        "^BSESN",     # BSE SENSEX (India)
        "^KS11",      # KOSPI (Korea)
        "^TWII",      # Taiwan Weighted
        "^MXX",       # IPC (Mexico)
        "^BVSP",      # BOVESPA (Brazil)
    ],
    "rates": [
        "^IRX",       # 13-week T-bill yield
        "^FVX",       # 5-year Treasury yield
        "^TNX",       # 10-year Treasury yield
        "^TYX",       # 30-year Treasury yield
    ],
    "currency": [
        "DX-Y.NYB",   # US Dollar Index (Yahoo's symbol for DXY)
    ],
}


def index_category_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for cat, tickers in INDEX_CATEGORIES.items():
        for t in tickers:
            out[t] = cat
    return out
