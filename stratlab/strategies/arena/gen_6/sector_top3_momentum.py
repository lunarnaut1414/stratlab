"""SPDR sector ETF top-3 momentum rotation.

Hypothesis: Rank all 9 core SPDR sector ETFs by 42-day return; hold top 3
equally weighted; rebalance every 10 bars (~2 weeks). No index gate or VIX
filter -- pure cross-sector relative momentum.

Structural distinctions vs existing leaderboard:
- Sector ETFs only (no broad SPY/QQQ + safe haven bond switching)
- No macro or trend gate (no SPY SMA, no VIX level)
- 9-way ranking concentrated to top-3 (novel pool width)
- 42-day window (not 63 or 126) for slightly faster momentum signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Core SPDR sector ETFs, all trading since 1998
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
# XLC only starts 2018 (misses most of IS), so we'll compute scores for available ETFs

MOMENTUM_WINDOW = 42
TOP_K = 3
REBALANCE_EVERY = 10
EXPOSURE = 0.97


class SectorTop3Momentum(Strategy):
    """Sector ETF top-3 momentum rotation without any macro gate."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            top_k=top_k,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 5
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

        # Compute momentum for each sector ETF
        scores: dict[str, float] = {}
        for sym in SECTOR_ETFS:
            try:
                hist = ctx.history(sym)
            except Exception:
                continue
            if len(hist) < self.momentum_window + 2:
                continue
            closes = hist["close"].dropna()
            if len(closes) < self.momentum_window:
                continue
            ret = float(closes.iloc[-1] / closes.iloc[-self.momentum_window] - 1.0)
            if np.isfinite(ret):
                scores[sym] = ret

        if len(scores) < self.top_k:
            return []

        # Select top-K by momentum
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        longs = ranked[: self.top_k]
        per_weight = self.exposure / len(longs)
        target = {sym: per_weight for sym in longs}

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


NAME = "sector_top3_momentum"
HYPOTHESIS = (
    "SPDR sector ETF top-3 momentum rotation: rank all 9 core SPDR sector ETFs "
    "by 42-day return, hold top-3 equally, rebalance every 10 bars; "
    "no index gate; captures cross-sector relative momentum"
)
UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SPY"]
STRATEGY = SectorTop3Momentum()
