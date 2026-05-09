"""Build the full default universe (S&P 500 + Nasdaq-100 + Dow 30 + ETFs +
inverse + leveraged) and pull 5 years of daily bars.

First run downloads from Yahoo (~2-5 min for ~700 tickers).
Subsequent runs read from ~/.stratlab/cache/.
"""
from __future__ import annotations

import pandas as pd

from stratlab import (
    default_universe,
    dow30_tickers,
    inverse_etfs,
    leveraged_etfs,
    load_universe,
    nasdaq100_tickers,
    popular_etfs,
    sp500_tickers,
)


def main() -> None:
    sp500 = sp500_tickers()
    ndx = nasdaq100_tickers()
    dow = dow30_tickers()
    etfs = popular_etfs()
    inv = inverse_etfs()
    lev = leveraged_etfs()
    full = default_universe()

    print("Universe breakdown:")
    print(f"  S&P 500       : {len(sp500)}")
    print(f"  Nasdaq-100    : {len(ndx)}")
    print(f"  Dow 30        : {len(dow)}")
    print(f"  Popular ETFs  : {len(etfs)}")
    print(f"  Inverse ETFs  : {len(inv)}")
    print(f"  Leveraged ETFs: {len(lev)}")
    print(f"  ---")
    print(f"  Default (deduped): {len(full)}")

    print(f"\nLoading 5y of daily bars for {len(full)} tickers...")
    bars = load_universe(full, start="2020-01-01")
    print(f"  Got data for {len(bars)} / {len(full)} tickers")

    failed = sorted(set(full) - set(bars.keys()))
    if failed:
        print(f"  No data for {len(failed)}: {failed[:20]}{' ...' if len(failed) > 20 else ''}")

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

    print("\nBars per ticker — distribution:")
    print(summary["bars"].describe())


if __name__ == "__main__":
    main()
