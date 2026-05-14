"""Multi-Asset Absolute Momentum Rotation — gen_7 sonnet-7 (attempt 8)

Hypothesis: Each month rank 8 diverse ETFs (SPY, QQQ, TLT, IEF, GLD, XLE,
EEM, LQD) by 3-month total return. Hold the top-3 that have POSITIVE absolute
momentum (3m return > 0). For any slot without a positive-momentum asset,
hold SHY (cash proxy).

Rationale: Gary Antonacci's absolute momentum work shows that holding assets
only when they have positive 3m absolute momentum significantly improves
risk-adjusted returns vs pure relative momentum. The diverse universe
(equity, bonds, gold, energy, EM, credit) means the strategy naturally shifts
between asset classes based on what's working.

Key distinctions from leaderboard:
- Diverse universe includes EM (EEM), credit (LQD), energy (XLE) - novel mix
- Absolute momentum filter prevents holding declining assets
- No VIX gate, no credit gate - pure price-momentum driven
- Monthly rebalance with 3-month signal (21 bars)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "GLD", "XLE", "EEM", "LQD", "SHY"]

REBALANCE_EVERY = 21       # monthly
MOMENTUM_WINDOW = 63       # 3-month
TOP_K = 3                  # hold top 3
EXPOSURE = 0.97
SHY = "SHY"
CANDIDATES = ["SPY", "QQQ", "TLT", "IEF", "GLD", "XLE", "EEM", "LQD"]


class MultiAssetAbsoluteMomentum(Strategy):
    """Multi-asset absolute momentum: top-3 positive-momentum ETFs from diverse universe."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute 3-month momentum for all candidate ETFs
        prices = ctx.closes_window(self.momentum_window + 5)
        if len(prices) < self.momentum_window:
            return []

        scores: dict[str, float] = {}
        for sym in CANDIDATES:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-self.momentum_window])
            if p_start <= 0:
                continue
            ret = p_end / p_start - 1.0
            if np.isfinite(ret):
                scores[sym] = ret

        # Absolute momentum filter: only positive returns qualify
        positive_scores = {sym: ret for sym, ret in scores.items() if ret > 0}

        target: dict[str, float] = {}

        if not positive_scores:
            # All assets declining: full SHY
            if SHY in live:
                target[SHY] = self.exposure
        else:
            # Rank by momentum, take top-K with positive momentum
            k = min(self.top_k, len(positive_scores))
            ranked = sorted(positive_scores, key=positive_scores.__getitem__, reverse=True)[:k]
            per_weight = self.exposure / self.top_k  # keep equal slots even if fewer positive

            for sym in ranked:
                if sym in live:
                    target[sym] = per_weight

            # Fill remaining slots with SHY if < top_k positive
            remaining_slots = self.top_k - len(ranked)
            if remaining_slots > 0 and SHY in live:
                shy_weight = per_weight * remaining_slots
                target[SHY] = target.get(SHY, 0.0) + shy_weight

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


NAME = "multiasset_absolute_momentum"
HYPOTHESIS = (
    "Multi-asset absolute momentum: each month rank SPY/QQQ/TLT/IEF/GLD/XLE/EEM/LQD by "
    "3-month return; hold top-3 with positive absolute momentum equal-weight; fill remaining "
    "slots with SHY; purely price-momentum driven without VIX or credit gates; diverse "
    "asset class coverage including EM and energy"
)

STRATEGY = MultiAssetAbsoluteMomentum()
