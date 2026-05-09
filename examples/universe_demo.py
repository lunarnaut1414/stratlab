"""Fetch the current S&P 500 and pull 5 years of daily bars.

First run downloads from Yahoo (~1-2 min for 500 tickers).
Subsequent runs read from ~/.stratlab/cache/.
"""
from __future__ import annotations

import pandas as pd

from stratlab import load_universe, sp500_tickers


def main() -> None:
    tickers = sp500_tickers()
    print(f"Universe size: {len(tickers)} tickers")
    print(f"First 10: {tickers[:10]}")

    bars = load_universe(tickers, start="2020-01-01")
    print(f"\nLoaded {len(bars)} / {len(tickers)} tickers with data")

    failed = sorted(set(tickers) - set(bars.keys()))
    if failed:
        print(f"No data for: {failed}")

    summary = pd.DataFrame(
        {
            sym: {
                "start": df.index.min().date(),
                "end": df.index.max().date(),
                "bars": len(df),
                "last_close": df["close"].iloc[-1],
            }
            for sym, df in bars.items()
        }
    ).T

    print("\nSample (first 5 tickers):")
    print(summary.head())

    print("\nBars per ticker — distribution:")
    print(summary["bars"].describe())


if __name__ == "__main__":
    main()
