"""Ticker → category catalog.

Drives the on-disk folder layout (``stocks/<gics_sector>/...``,
``etfs/<category>/...``) and is persisted to ``catalog.json`` in the market
data directory so it can be inspected without re-scraping. The catalog is
authoritative for category lookup; if a ticker isn't in it, callers fall back
to ``uncategorized`` rather than guessing.

Stock sectors come from the same Wikipedia page used for the S&P 500 ticker
list — the ``GICS Sector`` column. ETF categories come from ``_etf_lists.py``.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from stratlab.data._etf_lists import ETF_CATEGORIES, etf_category_map

_USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

CATALOG_VERSION = 1
UNCATEGORIZED = "uncategorized"


def _slugify(name: str) -> str:
    """Turn a human GICS sector name into a folder-safe lowercase slug.

    ``"Information Technology"`` → ``"information_technology"``,
    ``"Health Care"`` → ``"health_care"``,
    ``"Communication Services"`` → ``"communication_services"``.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def scrape_sp500_sectors() -> dict[str, str]:
    """Fetch ``{ticker: gics_sector_slug}`` from the S&P 500 Wikipedia page."""
    resp = requests.get(SP500_WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    chosen = None
    for table in tables:
        if "Symbol" in table.columns and "GICS Sector" in table.columns:
            chosen = table
            break
    if chosen is None:
        raise ValueError("Could not find S&P 500 constituents table with GICS Sector column")

    sectors: dict[str, str] = {}
    for _, row in chosen.iterrows():
        ticker = str(row["Symbol"]).replace(".", "-").strip()
        sector = str(row["GICS Sector"]).strip()
        if ticker and sector and sector.lower() != "nan":
            sectors[ticker] = _slugify(sector)
    return sectors


def build_catalog() -> dict:
    """Assemble the full catalog dict from Wikipedia + ETF lists.

    Structure::

        {
          "version": 1,
          "generated_at": "<isoformat>",
          "stocks": {"AAPL": {"sector": "information_technology"}, ...},
          "etfs":   {"SPY": {"category": "broad_market"}, ...},
        }
    """
    stock_sectors = scrape_sp500_sectors()
    etf_map = etf_category_map()

    return {
        "version": CATALOG_VERSION,
        "generated_at": datetime.now().isoformat(),
        "stocks": {ticker: {"sector": sector} for ticker, sector in stock_sectors.items()},
        "etfs": {ticker: {"category": cat} for ticker, cat in etf_map.items()},
    }


def save_catalog(catalog: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2, sort_keys=True))


def load_catalog(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def category_for(ticker: str, catalog: dict | None) -> str:
    """Return the path-friendly subfolder for ``ticker``.

    Returns one of:

    - ``"stocks/<sector_slug>"`` if the ticker is a known stock
    - ``"etfs/<category>"`` if the ticker is a known ETF
    - ``"uncategorized"`` otherwise

    The catalog is queried in-memory; pass ``None`` to disable lookups (useful
    for paths that should bypass categorization, like when migrating).
    """
    if catalog is None:
        return UNCATEGORIZED

    stock = catalog.get("stocks", {}).get(ticker)
    if stock and stock.get("sector"):
        return f"stocks/{stock['sector']}"

    etf = catalog.get("etfs", {}).get(ticker)
    if etf and etf.get("category"):
        return f"etfs/{etf['category']}"

    return UNCATEGORIZED


def all_etf_categories() -> list[str]:
    """Names of every ETF category in the curated lists."""
    return list(ETF_CATEGORIES.keys())
