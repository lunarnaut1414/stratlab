"""Factor ETF rotation strategy.

Hypothesis: Rank MTUM/VLUE/VTV/VUG by 3-month (63d) total return;
hold top 2 equally weighted; rebalance monthly (every 21 bars).
No VIX gate, no trend filter -- pure factor momentum rotation.

Structural distinctions vs existing leaderboard:
- Factor ETFs only (not sector ETFs or broad market + safe haven)
- No macro gate (VIX / SPY SMA absent)
- Uses 4 distinct factor tilts (momentum, value x2, growth)
- MTUM/VLUE available from ~2013; VTV/VUG from 2004 for full IS window
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FACTOR_ETFS = ["MTUM", "VLUE", "VTV", "VUG"]
CASH_PROXY = "IEF"
MOMENTUM_WINDOW = 63   # 3 months
TOP_K = 2
REBALANCE_EVERY = 21   # monthly
EXPOSURE = 0.97


class FactorETFRotation(Strategy):
    """Factor ETF momentum rotation: hold top-2 factor ETFs by 3-month return."""

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

        # Compute momentum for each factor ETF with available data
        scores: dict[str, float] = {}
        for sym in FACTOR_ETFS:
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

        target: dict[str, float] = {}

        if len(scores) >= 1:
            # Hold top-K by momentum, or all if fewer than K available
            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
            per_weight = self.exposure / k
            for sym in ranked:
                target[sym] = per_weight
        else:
            # Fallback: hold cash proxy
            if CASH_PROXY in closes_now.index:
                target[CASH_PROXY] = self.exposure

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


NAME = "factor_etf_rotation"
HYPOTHESIS = (
    "Factor ETF rotation: rank MTUM/VLUE/VTV/VUG by 3-month total return; "
    "hold top 2 equally weighted; rebalance monthly; no VIX or trend gate"
)
UNIVERSE = ["MTUM", "VLUE", "VTV", "VUG", "IEF", "SPY"]
STRATEGY = FactorETFRotation()
