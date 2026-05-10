"""Tech vs Defensive Sector Rotation — gen_5 sonnet-5

Hypothesis: Tech vs defensive sector rotation — rank XLK vs XLU by 60-day
momentum and hold the stronger sector ETF in full. Switch when the leader
changes, with a 5-day minimum hold to avoid churn.

Rationale: XLK (technology) and XLU (utilities/defensive) exhibit strong
cyclical divergence. Technology leads in risk-on environments; utilities
lead in risk-off/rate-sensitive environments. A simple momentum signal
over 60 days captures this rotation cleanly.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["XLK", "XLU"]

MOMENTUM_WINDOW = 60
MIN_HOLD_DAYS = 5
EXPOSURE = 0.97

_ETF_A = "XLK"
_ETF_B = "XLU"


class TechVsDefensiveRotation(Strategy):
    """Rank XLK vs XLU by 60d momentum; hold the stronger ETF fully."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)
        self._current_holding: str | None = None
        self._days_in_position: int = 0

    def on_start(self) -> None:
        self._current_holding = None
        self._days_in_position = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Need enough history for momentum window
        hist_a = ctx.history(_ETF_A)
        hist_b = ctx.history(_ETF_B)

        if len(hist_a) < MOMENTUM_WINDOW + 1 or len(hist_b) < MOMENTUM_WINDOW + 1:
            return []

        closes = ctx.closes()
        if _ETF_A not in closes or _ETF_B not in closes:
            return []

        price_a = float(closes[_ETF_A])
        price_b = float(closes[_ETF_B])
        if price_a <= 0 or price_b <= 0:
            return []

        # Compute 60-day momentum for each ETF
        close_a = hist_a["close"]
        close_b = hist_b["close"]

        start_a = float(close_a.iloc[-MOMENTUM_WINDOW])
        start_b = float(close_b.iloc[-MOMENTUM_WINDOW])

        if start_a <= 0 or start_b <= 0:
            return []

        mom_a = float(close_a.iloc[-1]) / start_a - 1.0
        mom_b = float(close_b.iloc[-1]) / start_b - 1.0

        # Determine target based on momentum
        target_sym = _ETF_A if mom_a >= mom_b else _ETF_B
        target_price = price_a if target_sym == _ETF_A else price_b

        # Enforce minimum hold days to avoid churn
        if self._current_holding is not None:
            self._days_in_position += 1

        if self._current_holding == target_sym:
            # Already holding the right one — check sizing
            current_pos = ctx.position(target_sym)
            if current_pos.size > 0:
                return []  # Already positioned, nothing to do

        # Check minimum hold constraint
        if (self._current_holding is not None
                and self._current_holding != target_sym
                and self._days_in_position < MIN_HOLD_DAYS):
            return []  # Too soon to switch

        orders: list[Order] = []

        # Close any existing position
        for sym in [_ETF_A, _ETF_B]:
            pos = ctx.position(sym)
            if pos.size > 0 and sym != target_sym:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        # Open new position if not already there
        current = ctx.position(target_sym).size
        # Estimate available capital
        equity = ctx.cash
        for sym in [_ETF_A, _ETF_B]:
            p = ctx.position(sym)
            if p.size > 0:
                price = price_a if sym == _ETF_A else price_b
                equity += p.size * price

        target_shares = int(equity * EXPOSURE / target_price)
        if target_shares > current:
            delta = target_shares - current
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))

        if orders:
            self._current_holding = target_sym
            self._days_in_position = 0

        return orders


NAME = "tech_vs_defensive_rotation"
HYPOTHESIS = (
    "Rank XLK vs XLU by 60-day momentum and hold the stronger sector ETF in full; "
    "switch when leader changes, with 5-day minimum hold to avoid churn."
)

STRATEGY = TechVsDefensiveRotation()
