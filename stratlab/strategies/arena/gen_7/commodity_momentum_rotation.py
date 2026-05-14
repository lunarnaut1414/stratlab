"""Commodity momentum rotation: DBC / GLD / USO / PDBC monthly momentum.

Hypothesis: each month rank commodity ETFs by 3-month return, hold top-2
equal-weight when both have positive absolute momentum, else SHY for any
slot with negative 3-month return. Purely commodity-focused, uncorrelated
to equity momentum strategies that dominate the leaderboard.

Rationale:
  - Commodities have a well-documented momentum premium (Gorton & Rouwenhorst
    2006, Asness et al 2013) that is structurally different from equity momentum.
  - The DBC/GLD/USO/PDBC universe covers broad commodities, gold, crude oil, and
    diversified commodities (PDBC = no-K1 alternative to DBC).
  - Absolute momentum filter (only hold when 3m return > 0) avoids deep commodity
    drawdowns during deflation/contraction regimes.
  - SHY cash proxy when no commodity passes the filter provides downside protection.
  - Monthly rebalance (21 bars) is appropriate for commodity momentum signals.

Distinction from existing strategies:
  - All other accepted strategies use equity (SP500 stocks, QQQ, SPY) or
    bond/credit signals as their primary building blocks. This is the only
    pure-commodity allocator.
  - Low expected correlation to nearhi_momentum_quality, ensemble, and
    bond_termstruct strategies.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["DBC", "GLD", "USO", "PDBC", "SHY"]

REBALANCE_EVERY = 21       # monthly
MOMENTUM_WINDOW = 63       # 3-month
EXPOSURE = 0.97
TOP_K = 2
COMMODITY_SYMBOLS = ["DBC", "GLD", "USO", "PDBC"]


class CommodityMomentumRotation(Strategy):
    """Commodity momentum rotation across DBC/GLD/USO/PDBC with absolute filter."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        exposure: float = EXPOSURE,
        top_k: int = TOP_K,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            exposure=exposure,
            top_k=top_k,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.exposure = float(exposure)
        self.top_k = int(top_k)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 10
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

        # Compute 3-month momentum for each commodity ETF
        prices = ctx.closes_window(self.momentum_window + 5)
        if len(prices) < self.momentum_window:
            return []

        scores: dict[str, float] = {}
        for sym in COMMODITY_SYMBOLS:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-self.momentum_window])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            if np.isfinite(ret):
                scores[sym] = ret

        # Absolute momentum filter: only hold if positive return
        positive = {sym: ret for sym, ret in scores.items() if ret > 0}

        target: dict[str, float] = {}

        if not positive:
            # No commodities with positive momentum — go to SHY
            if "SHY" in closes_now.index:
                target["SHY"] = self.exposure
        else:
            # Rank by momentum, hold top-K
            k = min(self.top_k, len(positive))
            ranked = sorted(positive, key=positive.__getitem__, reverse=True)[:k]
            per_weight = self.exposure / len(ranked)
            for sym in ranked:
                if sym in closes_now.index:
                    target[sym] = per_weight
            # If some slots left without positive commodities, fill with SHY
            if len(ranked) < self.top_k and "SHY" in closes_now.index:
                remaining = self.top_k - len(ranked)
                target["SHY"] = target.get("SHY", 0) + per_weight * remaining

        orders: list[Order] = []

        # Liquidate positions not in target
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


NAME = "commodity_momentum_rotation"
HYPOTHESIS = (
    "Commodity momentum rotation DBC/GLD/USO/PDBC: each month rank commodity ETFs by "
    "3-month return, hold top-2 equal-weight when positive absolute momentum, else SHY; "
    "purely commodity-focused uncorrelated to equity momentum leaderboard"
)

STRATEGY = CommodityMomentumRotation()
