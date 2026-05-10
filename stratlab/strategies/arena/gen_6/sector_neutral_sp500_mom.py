"""Sector-neutral SP500 momentum strategy.

Hypothesis: Pick top-3 SP500 stocks by 63-day momentum within each of 5
main GICS sectors (technology, healthcare, financials, industrials, consumer
discretionary). Hold the resulting 15 stocks equally weighted. SPY 200d SMA
gate to defensives (TLT) in bear markets. Rebalance every 10 bars.

Structural distinctions:
- Sector-neutral construction prevents concentration in single sector
  (unlike pure momentum which piles into tech during tech rallies)
- 5 sectors * 3 picks = 15 holdings with forced diversification
- Different daily return path from single-sorted SP500 momentum
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 63
PICKS_PER_SECTOR = 3
REBALANCE_EVERY = 10
TREND_WINDOW = 200
EXPOSURE = 0.97

# GICS sector classification for SP500 members
# Using ticker lists for the 5 main sectors
# We'll use SPDR sector ETFs to identify which sector is bullish,
# then pick top stocks from the SP500 universe by sector label stored
# in the data catalog

# Since we don't have direct sector labels in the strategy context,
# use sector ETFs as proxies to identify sector membership
# Key: pick top stocks from the SP500 universe that have high momentum,
# then use sector ETF baskets to ensure rough sector balance

# Sectors and their representative ETF sets (used to infer rough sector composition)
SECTOR_ETFS = {
    "tech": ["XLK"],
    "health": ["XLV"],
    "finance": ["XLF"],
    "industrial": ["XLI"],
    "consumer_disc": ["XLY"],
}

# Defensive rotation targets
DEFENSIVE = {"TLT": 0.6, "SHY": 0.37}


class SectorNeutralSP500Mom(Strategy):
    """Sector-neutral SP500 momentum: top-3 per sector x 5 sectors, SPY-gated."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        picks_per_sector: int = PICKS_PER_SECTOR,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            picks_per_sector=picks_per_sector,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.picks_per_sector = int(picks_per_sector)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def _get_sector_label(self, sym: str, sector_comps: dict[str, list[str]]) -> str | None:
        """Map symbol to sector using the pre-built sector_comps dict."""
        for sector, members in sector_comps.items():
            if sym in members:
                return sector
        return None

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history("SPY")
            spy_closes = spy_hist["close"].dropna()
            bull = float(spy_closes.iloc[-1]) > float(spy_closes.iloc[-self.trend_window:].mean())
        except Exception:
            bull = True

        target: dict[str, float] = {}

        if not bull:
            for sym, wt in DEFENSIVE.items():
                if sym in live:
                    target[sym] = wt * self.exposure
        else:
            # Get momentum for all SP500 stocks
            need = self.momentum_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            # Compute momentum scores
            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                return []

            # Get sector ETF momentum to determine which sectors are leading
            sector_moms: dict[str, float] = {}
            for sector, etfs in SECTOR_ETFS.items():
                sector_rets = []
                for etf in etfs:
                    try:
                        etf_hist = ctx.history(etf)
                        etf_c = etf_hist["close"].dropna()
                        if len(etf_c) >= self.momentum_window:
                            r = float(etf_c.iloc[-1] / etf_c.iloc[-self.momentum_window] - 1.0)
                            sector_rets.append(r)
                    except Exception:
                        pass
                if sector_rets:
                    sector_moms[sector] = float(np.mean(sector_rets))

            # Assign stocks to sectors based on correlation with sector ETFs
            # As a proxy: use correlation of 40d returns to each sector ETF
            # Build sector ETF return series
            sector_etf_rets: dict[str, pd.Series] = {}
            for sector, etfs in SECTOR_ETFS.items():
                for etf in etfs:
                    try:
                        etf_hist = ctx.history(etf)
                        etf_c = etf_hist["close"].dropna().iloc[-45:]
                        if len(etf_c) >= 20:
                            r = etf_c.pct_change().dropna()
                            sector_etf_rets[sector] = r
                            break
                    except Exception:
                        pass

            # For each stock, find its sector by highest return correlation
            stock_sector: dict[str, str] = {}
            # Only process stocks in the SP500-like universe (filter by available data)
            # To keep it simple and fast: assign based on simple correlation
            stock_syms = [s for s in scores if s in prices.columns]

            if sector_etf_rets:
                for sym in stock_syms:
                    col = prices[sym].dropna().iloc[-45:]
                    if len(col) < 20:
                        continue
                    stock_r = col.pct_change().dropna()
                    best_corr = -2.0
                    best_sec = "tech"  # default
                    for sec, sec_r in sector_etf_rets.items():
                        aligned = sec_r.align(stock_r, join="inner")[0]
                        stock_a = sec_r.align(stock_r, join="inner")[1]
                        if len(aligned) < 10:
                            continue
                        corr = float(np.corrcoef(aligned.values, stock_a.values)[0, 1])
                        if np.isfinite(corr) and corr > best_corr:
                            best_corr = corr
                            best_sec = sec
                    stock_sector[sym] = best_sec
            else:
                # No sector ETF data, just use all as "other"
                for sym in stock_syms:
                    stock_sector[sym] = "tech"

            # Pick top picks_per_sector stocks from each sector
            sector_buckets: dict[str, list[tuple[str, float]]] = {s: [] for s in SECTOR_ETFS}
            for sym, ret in scores.items():
                sec = stock_sector.get(sym, "tech")
                sector_buckets[sec].append((sym, ret))

            longs = []
            for sec, bucket in sector_buckets.items():
                if not bucket:
                    continue
                bucket_sorted = sorted(bucket, key=lambda x: x[1], reverse=True)
                picks = bucket_sorted[: self.picks_per_sector]
                longs.extend([sym for sym, _ in picks])

            if not longs:
                return []

            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SHY", "SPY", "XLK", "XLV", "XLF", "XLI", "XLY"]


NAME = "sector_neutral_sp500_mom"
HYPOTHESIS = (
    "Sector-first stock momentum: identify top-2 SPDR sectors by 42d momentum "
    "(SPY 200d SMA gate), then hold top-10 SP500 stocks from those winning sectors "
    "by 21d momentum; equal-weight; rebalance every 10 bars"
)
UNIVERSE = _universe
STRATEGY = SectorNeutralSP500Mom()
