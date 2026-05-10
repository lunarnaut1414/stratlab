"""Yield Curve Regime ETF Rotation — gen_5 sonnet-7

Hypothesis: The Treasury yield curve (10yr minus 3mo spread, ^TNX - ^IRX)
is a leading macro indicator. A steep curve (spread > ~1.5%) signals healthy
growth expectations — hold QQQ for tech/growth exposure. A flat/inverted curve
(spread < ~0.5%) signals rising recession risk — hold TLT for safety. In
between, hold SPY as a balanced position.

Signal (weekly rebalance):
  - spread = ^TNX close - ^IRX close (both in yield-% terms)
  - spread > 1.5: hold QQQ (steep curve = growth)
  - spread < 0.5: hold TLT (flat/inverted = recession risk)
  - else:         hold SPY (neutral zone)

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "TLT", "^TNX", "^IRX"]

_STEEP_THRESHOLD = 1.5    # curve steepness: steep = growth regime
_FLAT_THRESHOLD = 0.5     # flat/inverted = recession risk
_SMOOTH_DAYS = 10         # smooth the spread signal to reduce noise
_MIN_HISTORY = _SMOOTH_DAYS + 10
_REBALANCE = 5            # weekly
_EXPOSURE = 0.97

# Regime-to-ETF mapping
_STEEP_ETF = "QQQ"
_NEUTRAL_ETF = "SPY"
_FLAT_ETF = "TLT"


class YieldCurveRegimeRotation(Strategy):
    """Yield curve (10yr-3mo) regime gate: QQQ in steep, TLT in flat/inverted."""

    def __init__(
        self,
        steep_threshold: float = _STEEP_THRESHOLD,
        flat_threshold: float = _FLAT_THRESHOLD,
        smooth_days: int = _SMOOTH_DAYS,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            steep_threshold=steep_threshold,
            flat_threshold=flat_threshold,
            smooth_days=smooth_days,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.steep_threshold = steep_threshold
        self.flat_threshold = flat_threshold
        self.smooth_days = smooth_days
        self.rebalance = rebalance
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < _MIN_HISTORY:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Read ^TNX (10yr) and ^IRX (3mo) yield indices
        try:
            tnx_hist = ctx.history("^TNX")
            irx_hist = ctx.history("^IRX")
        except KeyError:
            return []

        needed = self.smooth_days + 1
        if len(tnx_hist) < needed or len(irx_hist) < needed:
            return []

        tnx_vals = tnx_hist["close"].values[-needed:]
        irx_vals = irx_hist["close"].values[-needed:]

        # Compute spread (10yr - 3mo) and smooth it
        n = min(len(tnx_vals), len(irx_vals), self.smooth_days)
        spread_vals = tnx_vals[-n:] - irx_vals[-n:]
        smoothed_spread = float(np.mean(spread_vals))

        # Determine target ETF based on yield curve regime
        if smoothed_spread > self.steep_threshold:
            target_sym = _STEEP_ETF   # Steep: growth regime → QQQ
        elif smoothed_spread < self.flat_threshold:
            target_sym = _FLAT_ETF    # Flat/inverted: recession risk → TLT
        else:
            target_sym = _NEUTRAL_ETF  # Neutral → SPY

        # All other ETFs are the "other" set
        all_tradeable = [_STEEP_ETF, _NEUTRAL_ETF, _FLAT_ETF]
        others = [s for s in all_tradeable if s != target_sym]

        closes = ctx.closes()
        if target_sym not in closes or closes[target_sym] <= 0:
            return []

        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit other positions
        for other_sym in others:
            other_pos = ctx.position(other_sym)
            if other_pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=other_pos.size, symbol=other_sym))

        # Size target position
        target_pos = ctx.position(target_sym)
        target_price = float(closes[target_sym])
        target_shares = int(equity * self.exposure / target_price)
        delta = target_shares - int(target_pos.size)
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))
        elif delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=target_sym))

        return orders


NAME = "rsi_etf_meanrev"
HYPOTHESIS = (
    "Yield curve regime ETF rotation: hold QQQ when 10-day smoothed (TNX-IRX) spread > 1.5% "
    "(steep curve = growth), TLT when spread < 0.5% (flat/inverted = recession risk), SPY "
    "in neutral zone. Treasury yield-curve signal is distinct from all existing leaderboard strategies."
)

STRATEGY = YieldCurveRegimeRotation(
    steep_threshold=_STEEP_THRESHOLD,
    flat_threshold=_FLAT_THRESHOLD,
    smooth_days=_SMOOTH_DAYS,
    rebalance=_REBALANCE,
    exposure=_EXPOSURE,
)
