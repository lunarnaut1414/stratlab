"""Popular ETF Cross-Sectional Absolute Momentum — gen_8 sonnet-10

Hypothesis: Each month, rank all popular_etfs universe ETFs by 63-day return.
Hold the top-5 ETFs that have positive absolute 63-day momentum (return > 0).
For any unfilled slot (if fewer than 5 qualify), hold SHY.

Equal-weight. Monthly rebalance (21 bars).

Rationale: Using the popular_etfs universe (broad ETFs spanning sectors,
geographies, bonds, commodities, gold, etc.) provides natural diversification.
The absolute-momentum filter ensures we only hold ETFs in actual uptrend —
avoiding the "hold momentum even when everything is falling" problem.
Cross-sectional ETF momentum is structurally different from SP500 stock selection
because:
  1. ETFs span multiple asset classes (equity, bond, commodity, international)
  2. The winner ETFs shift across asset classes through market cycles
  3. Much lower turnover than stock-level strategies

This captures the time-series (absolute) momentum property AND cross-sectional
ranking, but at the ETF level rather than individual stock level.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21      # monthly
MOMENTUM_WINDOW = 63      # ~3 months
TOP_K = 5
EXPOSURE = 0.97
_SHY = "SHY"

# ETFs to exclude from ranking (use only as safe-haven slots)
_EXCLUDE = {"SHY", "BIL", "SHV"}


class PopularETFAbsMomentum(Strategy):
    """Top-5 popular ETFs by 63d return with absolute momentum filter."""

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

        # Get momentum window prices
        need = self.momentum_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < self.momentum_window:
            return []

        # Rank by 63d return (absolute momentum filter: must be positive)
        scores: dict[str, float] = {}
        for sym in prices.columns:
            if sym in _EXCLUDE:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window:
                continue
            ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
            if np.isfinite(ret) and ret > 0 and sym in live:
                scores[sym] = ret

        # Select top-K with positive momentum
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        selected_risk = ranked[:self.top_k]

        # Build target: risk ETFs + SHY for unfilled slots
        target: dict[str, float] = {}
        per_slot = self.exposure / self.top_k

        for sym in selected_risk:
            target[sym] = per_slot

        # Fill remaining slots with SHY
        n_remaining = self.top_k - len(selected_risk)
        if n_remaining > 0 and _SHY in live:
            target[_SHY] = target.get(_SHY, 0.0) + per_slot * n_remaining

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


UNIVERSE = "popular_etfs"

NAME = "popular_etf_abs_momentum"
HYPOTHESIS = (
    "Popular ETF cross-sectional 3-month absolute momentum: each month rank all popular_etfs "
    "ETFs by 63d return, hold top-5 that also have positive absolute 63d momentum (above zero); "
    "if fewer than 5 qualify hold SHY for unfilled slots; equal-weight; monthly rebalance; "
    "pure cross-sectional momentum on ETF universe with absolute filter avoids momentum stocks angle"
)

STRATEGY = PopularETFAbsMomentum()
