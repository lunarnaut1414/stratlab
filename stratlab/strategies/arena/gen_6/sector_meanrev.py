"""Sector ETF mean-reversion strategy.

Hypothesis:
  Buy the 3 worst-performing S&P 500 sector ETFs over the past 20 trading days
  when SPY is above its 200-day SMA (bull market gate).
  Equal-weight the 3 beaten-down sectors; hold 10 bars then re-rank.
  When SPY is below its 200-day SMA, rotate to SHY (cash-equivalent).

Rationale:
  Sector rotation frequently overshoots — the worst-performing sector over
  1-month intervals tends to mean-revert as capital flows back to depressed
  names. The 200d SMA gate prevents mean-reversion trades in structural
  downtrends where beaten sectors keep falling.

  This is structurally distinct from all existing leaderboard strategies
  (they all do momentum / trend-following / vol-gated equity, not
  cross-sector mean-reversion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# 11 US sector ETFs
SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLI", "XLP", "XLU",
    "XLE", "XLB", "XLRE", "XLY", "XLC",
]
REBALANCE_EVERY = 10   # 2 weeks
LOOKBACK = 20          # 1-month return for ranking
TOP_LOSER_K = 3        # buy the 3 worst sectors
TREND_WINDOW = 200     # SPY 200d SMA
EXPOSURE = 0.97


class SectorMeanRev(Strategy):
    """Buy beaten-down sector ETFs in a bull market; cash otherwise."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        lookback: int = LOOKBACK,
        top_k: int = TOP_LOSER_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            lookback=lookback,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.lookback = int(lookback)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.lookback + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- Trend filter: SPY 200d SMA ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Risk-off: hold SHY
            if "SHY" in closes_now.index:
                target["SHY"] = self.exposure
        else:
            # Rank sectors by 20-day return (ascending = worst performers)
            prices = ctx.closes_window(self.lookback + 5)
            if len(prices) < self.lookback:
                return []

            scores: dict[str, float] = {}
            for sym in SECTOR_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.lookback:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.lookback] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                return []

            # Buy the k worst (mean-reversion)
            ranked = sorted(scores, key=scores.__getitem__)  # ascending
            longs = ranked[: self.top_k]
            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
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


NAME = "sector_meanrev"
HYPOTHESIS = (
    "Equity sector mean-reversion: buy the 3 worst-performing S&P500 sector ETFs "
    "(XLK,XLV,XLF,XLI,XLP,XLU,XLE,XLB,XLRE,XLY,XLC) over 20 days when SPY is "
    "above 200d SMA, equal-weight, 10-bar hold; rotate to SHY when SPY below 200d SMA."
)

UNIVERSE = SECTOR_ETFS + ["SHY", "SPY"]

STRATEGY = SectorMeanRev()
