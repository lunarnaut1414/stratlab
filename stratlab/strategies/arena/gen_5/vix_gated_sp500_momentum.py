"""VIX-gated SP500 momentum strategy.

Hypothesis: Use VIX level as a direct risk regime signal:
  - When VIX < 25 (calm market): hold top-15 SP500 stocks by 63-day momentum,
    equally weighted. Rebalance every 10 bars.
  - When VIX >= 25 (elevated risk): rotate to SHY 50% + TLT 50% (safe haven).

Rationale: VIX above 25 historically marks regimes of elevated uncertainty
where equity momentum degrades and drawdowns spike. The 63-day lookback
captures the intermediate-term momentum factor (not too short for noise, not
too long for stale signals). Rebalancing every 10 bars generates sufficient
trade count across the IS window.

Key distinction from gen5_bond_equity_regime (already accepted):
  - Uses VIX level directly instead of TLT/SPY ratio MA crossover
  - Adds SHY as 50% of defensive allocation (not just TLT+GLD)
  - Top-15 stocks vs top-10 (broader diversification)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10   # bars
MOMENTUM_WINDOW = 63   # ~3 months
VIX_THRESHOLD = 25.0   # switch to defensive above this
TOP_K = 15
EXPOSURE = 0.97
_VIX = "^VIX"


class VixGatedSP500Momentum(Strategy):
    """VIX-level-gated momentum rotation: top-15 SP500 stocks vs SHY+TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vix_threshold: float = VIX_THRESHOLD,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vix_threshold=vix_threshold,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = rebalance_every
        self.momentum_window = momentum_window
        self.vix_threshold = vix_threshold
        self.top_k = top_k
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get VIX level
        vix_level = float("nan")
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        defensive = np.isfinite(vix_level) and vix_level >= self.vix_threshold

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        target: dict[str, float] = {}

        if defensive:
            # Safe haven: SHY 50% + TLT 50%
            for sym, weight in [("SHY", 0.5), ("TLT", 0.5)]:
                if sym in closes_now.index:
                    target[sym] = weight * self.exposure
        else:
            # Risk-on: top-K momentum stocks
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                return []

            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            longs = ranked[:self.top_k]
            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SHY", "SPY", _VIX]


NAME = "vix_gated_sp500_momentum"
HYPOTHESIS = (
    "VIX-gated SP500 momentum: rebalance every 10 bars into top-15 SP500 stocks by 63-day "
    "return when VIX < 25; rotate to SHY 50% + TLT 50% when VIX >= 25."
)

UNIVERSE = _universe

STRATEGY = VixGatedSP500Momentum()
