"""Sector relative-momentum vs SPY strategy.

Hypothesis: Rank 9 SPDR sector ETFs (XLK/XLF/XLV/XLI/XLP/XLU/XLE/XLB/XLY)
by 21-day return minus SPY 21-day return (relative alpha vs market).
Hold top-3 sector ETFs equally weighted in bull markets (SPY above 200d SMA).
Rotate to TLT when bearish. Rebalance every 10 bars.

Rationale: Sector relative strength captures which areas of the economy are
outperforming regardless of overall market direction. By measuring return
relative to SPY, we isolate sector-specific alpha rather than overall market
momentum. This differs from absolute momentum (which can return no sectors
in a bear market) - we always hold the best 3 sectors in bull mode.

Structural differences from existing leaderboard:
- Relative return (alpha vs SPY) not absolute momentum
- 9-sector universe with 3 holdings at all times (in bull)
- 21d lookback is shorter than most SP500 momentum strategies (42d-126d)
- No VIX gate, no credit signal, no vol-targeting
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY"]
UNIVERSE = SECTORS + ["SPY", "TLT"]

LOOKBACK = 21         # 21-day relative momentum window
TREND_WINDOW = 200    # SPY 200d SMA
REBALANCE_EVERY = 10  # biweekly
TOP_K = 3             # hold top-3 sectors
EXPOSURE = 0.97


class SectorRelativeMomentum(Strategy):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            lookback=lookback,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.lookback = int(lookback)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.lookback, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend filter
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Compute SPY 21d return
            if len(spy_close) < self.lookback + 2:
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                spy_start = float(spy_close.iloc[-(self.lookback + 1)])
                spy_end = float(spy_close.iloc[-1])
                if spy_start > 0:
                    spy_ret = spy_end / spy_start - 1.0
                else:
                    spy_ret = 0.0

                # Compute relative momentum for each sector
                rel_returns: dict[str, float] = {}
                for sym in SECTORS:
                    try:
                        hist = ctx.history(sym)
                    except KeyError:
                        continue
                    if hist is None or len(hist) < self.lookback + 2:
                        continue
                    close = hist["close"].dropna()
                    if len(close) < self.lookback + 1:
                        continue
                    start = float(close.iloc[-(self.lookback + 1)])
                    end = float(close.iloc[-1])
                    if start > 0 and np.isfinite(start) and np.isfinite(end):
                        sect_ret = end / start - 1.0
                        rel_returns[sym] = sect_ret - spy_ret  # relative alpha

                if len(rel_returns) < self.top_k:
                    # Not enough sectors, fallback to SPY
                    if "SPY" in live:
                        target["SPY"] = self.exposure
                else:
                    # Hold top-K by relative momentum
                    ranked = sorted(rel_returns, key=rel_returns.__getitem__, reverse=True)
                    top_sectors = ranked[:self.top_k]
                    w = self.exposure / len(top_sectors)
                    for sym in top_sectors:
                        if sym in live:
                            target[sym] = w

                    if not target and "SPY" in live:
                        target["SPY"] = self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sector_relative_momentum"
HYPOTHESIS = (
    "Sector relative-momentum vs SPY: rank 9 SPDR sector ETFs by 21d return "
    "minus SPY 21d return (relative alpha); hold top-3 sector ETFs equally "
    "weighted; SPY 200d SMA gate; TLT defensive when bearish; biweekly rebalance; "
    "pure sector relative strength vs market orthogonal to VIX/credit/yield signals"
)

STRATEGY = SectorRelativeMomentum()
