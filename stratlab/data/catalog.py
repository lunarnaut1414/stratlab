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
from stratlab.data._futures_lists import FUTURES_CATEGORIES, futures_category_map
from stratlab.data._index_lists import INDEX_CATEGORIES, index_category_map

_USER_AGENT = "stratlab/0.1 (https://github.com/lunarnaut1414/stratlab) python-requests"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
IWM_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
)

CATALOG_VERSION = 4
UNCATEGORIZED = "uncategorized"

# IWM uses ``"Communication"`` where SP500/GICS uses ``"Communication
# Services"``. Without normalization we'd get two parallel folders
# (``stocks/communication`` and ``stocks/communication_services``) holding
# what should be the same sector — keep the SP500-derived spelling so
# existing folders stay authoritative.
_GICS_SECTOR_ALIASES: dict[str, str] = {
    "communication": "communication_services",
}


def _slugify(name: str) -> str:
    """Turn a human GICS sector name into a folder-safe lowercase slug.

    ``"Information Technology"`` → ``"information_technology"``,
    ``"Health Care"`` → ``"health_care"``,
    ``"Communication Services"`` → ``"communication_services"``.
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return _GICS_SECTOR_ALIASES.get(s.strip("_"), s.strip("_"))


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


def scrape_iwm_sectors() -> dict[str, str]:
    """Fetch ``{ticker: gics_sector_slug}`` for Russell 2000 from the iShares
    IWM holdings CSV. The CSV has a populated ``Sector`` column for ~1900
    constituents — the easiest free source for small-cap GICS sectors."""
    resp = requests.get(IWM_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    raw_lines = resp.text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(raw_lines) if line.startswith("Ticker,")), None
    )
    if header_idx is None:
        raise ValueError("Could not locate Ticker header in IWM holdings file")

    df = pd.read_csv(io.StringIO("\n".join(raw_lines[header_idx:])))

    sectors: dict[str, str] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).replace(".", "-").strip()
        sector_raw = str(row.get("Sector", "")).strip()
        if not ticker or not sector_raw or sector_raw.lower() in ("nan", "-"):
            continue
        if "cash" in sector_raw.lower() or "derivative" in sector_raw.lower():
            continue
        sectors[ticker] = _slugify(sector_raw)
    return sectors


def build_catalog() -> dict:
    """Assemble the full catalog dict from Wikipedia + IWM + curated lists.

    Structure::

        {
          "version": 4,
          "generated_at": "<isoformat>",
          "stocks":   {"AAPL":    {"sector":   "information_technology"}, ...},
          "etfs":     {"SPY":     {"category": "broad_market"}, ...},
          "indices":  {"^VIX":    {"category": "volatility"}, ...},
          "futures":  {"CL=F":    {"category": "energy"}, ...},
        }

    Stock sectors are sourced in priority order:

    1. **Wikipedia S&P 500 GICS** — the most authoritative free source
       (~503 names). Wins on ties.
    2. **iShares IWM holdings** — Russell 2000 GICS sectors (~1900 names).
       Adds small-cap coverage that's not in the SP500 Wiki page.
    3. **``"other"`` fallback** — any ticker present in our equity universe
       sources (Nasdaq listed / Russell 2000 / etc.) but not covered by
       (1) or (2) gets ``{"sector": "other"}`` so it lands in
       ``stocks/other/`` instead of ``uncategorized/``. The remaining gap
       is the ~2000 Nasdaq Composite stocks that aren't in the S&P 500
       or Russell 2000 — fixing that gap requires per-ticker Yahoo
       ``Ticker(sym).info`` lookups, which we don't do at build time.
    """
    sp500_sectors = scrape_sp500_sectors()
    try:
        iwm_sectors = scrape_iwm_sectors()
    except Exception:
        iwm_sectors = {}

    # SP500 wins on ties (more authoritative GICS labeling).
    stock_sectors: dict[str, str] = {**iwm_sectors, **sp500_sectors}

    # Fill the long tail with "other" so unsorted equities land in
    # stocks/other/ rather than uncategorized/. We pull from the same
    # universe sources that the data layer fetches from, so any ticker
    # that gets cached has a home.
    try:
        from stratlab.data.universe import (
            dow30_tickers,
            nasdaq100_tickers,
            nasdaq_listed_tickers,
            russell2000_tickers,
            sp500_tickers,
        )

        equity_pool: set[str] = set()
        for fn in (
            sp500_tickers, nasdaq100_tickers, nasdaq_listed_tickers,
            russell2000_tickers, dow30_tickers,
        ):
            try:
                equity_pool.update(fn())
            except Exception:
                continue
        for t in equity_pool - set(stock_sectors):
            stock_sectors[t] = "other"
    except ImportError:
        pass  # universe module not yet importable during partial setups

    etf_map = etf_category_map()
    index_map = index_category_map()
    futures_map = futures_category_map()

    return {
        "version": CATALOG_VERSION,
        "generated_at": datetime.now().isoformat(),
        "stocks": {ticker: {"sector": sector} for ticker, sector in stock_sectors.items()},
        "etfs": {ticker: {"category": cat} for ticker, cat in etf_map.items()},
        "indices": {ticker: {"category": cat} for ticker, cat in index_map.items()},
        "futures": {ticker: {"category": cat} for ticker, cat in futures_map.items()},
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

    - ``"stocks/<sector_slug>"`` (S&P 500 constituents)
    - ``"etfs/<category>"`` (curated ETF list)
    - ``"indices/<category>"`` (curated index list — VIX, SPX, etc.)
    - ``"futures/<category>"`` (curated continuous futures)
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

    idx = catalog.get("indices", {}).get(ticker)
    if idx and idx.get("category"):
        return f"indices/{idx['category']}"

    fut = catalog.get("futures", {}).get(ticker)
    if fut and fut.get("category"):
        return f"futures/{fut['category']}"

    return UNCATEGORIZED


def all_etf_categories() -> list[str]:
    """Names of every ETF category in the curated lists."""
    return list(ETF_CATEGORIES.keys())


def all_index_categories() -> list[str]:
    return list(INDEX_CATEGORIES.keys())


def all_futures_categories() -> list[str]:
    return list(FUTURES_CATEGORIES.keys())


def migrate_uncategorized(market_dir: Path, catalog: dict) -> dict[str, int]:
    """Walk ``<market_dir>/uncategorized/`` and re-home each cached file
    into the folder its (now-rebuilt) catalog entry points to.

    Files are *moved*, not copied — the source is removed once the target
    has been written. Files for tickers that are still uncategorized after
    rebuild are left in place. Returns a per-bucket count
    (``{"moved": N, "kept": M, "skipped": K}``).
    """
    src_dir = market_dir / UNCATEGORIZED
    if not src_dir.is_dir():
        return {"moved": 0, "kept": 0, "skipped": 0}

    moved = kept = skipped = 0
    for src in sorted(src_dir.glob("*.csv")):
        # Filenames are ``<TICKER>_<interval>.csv`` — split on first '_'
        stem = src.stem
        if "_" not in stem:
            skipped += 1
            continue
        ticker = stem.rsplit("_", 1)[0]
        category = category_for(ticker, catalog)
        if category == UNCATEGORIZED:
            kept += 1
            continue
        dst_dir = market_dir / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            # Shouldn't normally happen, but be defensive — keep the newer file.
            if src.stat().st_mtime > dst.stat().st_mtime:
                dst.unlink()
                src.rename(dst)
                moved += 1
            else:
                src.unlink()
                kept += 1
        else:
            src.rename(dst)
            moved += 1
    return {"moved": moved, "kept": kept, "skipped": skipped}


def main(argv: list[str] | None = None) -> int:
    """CLI for catalog rebuild + on-disk migration.

    ``rebuild`` regenerates ``catalog.json`` from current sources (Wikipedia
    S&P 500, IWM holdings, curated ETF/index/futures lists). ``migrate``
    walks the existing ``uncategorized/`` folder and moves each cached
    OHLCV file into the folder its rebuilt catalog entry points to.

    Typical post-universe-extension flow::

        python -m stratlab.data.catalog rebuild
        python -m stratlab.data.catalog migrate
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["rebuild", "migrate", "show"])
    args = parser.parse_args(argv)

    from stratlab.data.provider import (
        CATALOG_PATH, MARKET_DIR, _invalidate_catalog_cache,
    )

    if args.action == "rebuild":
        catalog = build_catalog()
        save_catalog(catalog, CATALOG_PATH)
        _invalidate_catalog_cache()
        print(
            f"[catalog] rebuilt v{catalog['version']} → {CATALOG_PATH}\n"
            f"  stocks   : {len(catalog['stocks'])}\n"
            f"  etfs     : {len(catalog['etfs'])}\n"
            f"  indices  : {len(catalog['indices'])}\n"
            f"  futures  : {len(catalog['futures'])}"
        )
        return 0

    if args.action == "migrate":
        catalog = load_catalog(CATALOG_PATH)
        if catalog is None:
            print("[catalog] no catalog.json — run `rebuild` first.")
            return 2
        result = migrate_uncategorized(MARKET_DIR, catalog)
        print(
            f"[catalog] migrated uncategorized/ files\n"
            f"  moved   : {result['moved']}\n"
            f"  kept    : {result['kept']} (still uncategorized)\n"
            f"  skipped : {result['skipped']} (malformed names)"
        )
        return 0

    if args.action == "show":
        catalog = load_catalog(CATALOG_PATH)
        if catalog is None:
            print("[catalog] no catalog.json")
            return 2
        from collections import Counter
        sectors = Counter(v["sector"] for v in catalog["stocks"].values())
        etf_cats = Counter(v["category"] for v in catalog["etfs"].values())
        print("stocks by sector:")
        for sec, n in sectors.most_common():
            print(f"  {sec:30s} {n}")
        print("\netfs by category:")
        for cat, n in etf_cats.most_common():
            print(f"  {cat:30s} {n}")
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
