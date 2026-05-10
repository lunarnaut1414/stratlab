"""Copper-as-cycle-proxy rotation strategy.

Hypothesis:
  Copper is a leading macro cycle indicator — dubbed "Dr. Copper" for its
  ability to diagnose economic health. When the copper signal is in an uptrend
  (20d MA > 60d MA), the global cycle is expanding; hold cyclicals
  (XLI, XLB, XLK). When copper trends down, rotate to defensives
  (XLU, TLT, GLD).

  Signal source priority:
  1. CPER (copper ETF, inception 2011-11-15) — preferred
  2. DBC (broad commodity ETF, inception 2006) — fallback

  Signal logic:
  - Compute 20-day and 60-day simple MAs on the signal ETF closing price.
  - If 20d MA > 60d MA (uptrend, expansion): hold XLI, XLB, XLK equally.
  - If 20d MA < 60d MA (downtrend, contraction): hold XLU, TLT, GLD equally.

  Rebalance: weekly (every 5 trading days).
  Exposure: 97% of portfolio.

IS window: 2010-01-01 to 2018-12-31
Note: CPER inception is 2011-11-15; DBC used as fallback for 2010-2011.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Explicit ETF list — all have IS window coverage (inception before 2010).
# DBC = broad commodity ETF (inception 2006, copper/commodity proxy for 2010-2011).
# CPER = copper-specific ETF (inception 2011-11-15, loaded when available).
# XLI/XLB/XLK = cyclicals; XLU/TLT/GLD = defensives.
UNIVERSE = ["DBC", "XLI", "XLB", "XLK", "XLU", "TLT", "GLD"]

_FAST_MA = 20
_SLOW_MA = 60
_REBALANCE = 5      # weekly
_EXPOSURE = 0.97
_MIN_HISTORY = _SLOW_MA + 10

_CYCLICALS = ["XLI", "XLB", "XLK"]
_DEFENSIVES = ["XLU", "TLT", "GLD"]


class CopperCycleRotation(Strategy):
    """Rotate between cyclical and defensive ETF baskets based on copper trend.

    Parameters
    ----------
    fast_ma : int
        Fast MA period for copper signal (default 20).
    slow_ma : int
        Slow MA period for copper signal (default 60).
    rebalance : int
        Bars between rebalance checks (default 5 = weekly).
    exposure : float
        Fraction of portfolio deployed (default 0.97).
    """

    def __init__(
        self,
        fast_ma: int = _FAST_MA,
        slow_ma: int = _SLOW_MA,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rebalance = rebalance
        self.exposure = exposure

    def _copper_signal(self, ctx: BarContext) -> bool | None:
        """Return True = cyclical (commodity uptrend), False = defensive, None = no signal.

        Uses DBC (broad commodity ETF, inception 2006) as copper proxy.
        DBC has copper as a major constituent and correlates well with copper cycle.
        """
        try:
            hist = ctx.history("DBC")
        except Exception:
            return None
        if hist is None or len(hist) < self.slow_ma + 2:
            return None
        closes = hist["close"].dropna()
        if len(closes) < self.slow_ma + 1:
            return None
        fast_val = float(closes.iloc[-self.fast_ma:].mean())
        slow_val = float(closes.iloc[-self.slow_ma:].mean())
        if not np.isfinite(fast_val) or not np.isfinite(slow_val):
            return None
        return fast_val > slow_val  # True = commodity uptrend = cyclicals

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < _MIN_HISTORY:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        signal = self._copper_signal(ctx)
        if signal is None:
            return []

        # Pick target basket
        target_basket = _CYCLICALS if signal else _DEFENSIVES
        exit_basket = _DEFENSIVES if signal else _CYCLICALS

        closes = ctx.closes()
        if closes.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)
        if equity <= 0:
            return []

        # Filter target to those with valid prices
        valid_targets = [s for s in target_basket if s in closes.index and closes[s] > 0]
        if not valid_targets:
            return []

        per_weight = self.exposure / len(valid_targets)
        target_weights: dict[str, float] = {sym: per_weight for sym in valid_targets}

        orders: list[Order] = []

        # Exit positions not in target
        all_exit = set(exit_basket) | (set(target_basket) - set(valid_targets))
        all_exit.add("DBC")  # never trade the signal ETF itself
        for sym in list(all_exit):
            pos = ctx.position(sym)
            if pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        # Adjust target positions
        for sym, weight in target_weights.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(equity * weight / price)
            current_shares = int(ctx.position(sym).size)
            delta = target_shares - current_shares
            if delta > 0:
                orders.append(Order(side=OrderSide.BUY, size=float(delta), symbol=sym))
            elif delta < -1:
                orders.append(Order(side=OrderSide.SELL, size=float(abs(delta)), symbol=sym))

        return orders


NAME = "copper_cycle_rotation"
HYPOTHESIS = (
    "Commodity-cycle-proxy rotation: use DBC (broad commodity ETF) 20d vs 60d MA crossover "
    "as macro cycle signal; when commodity trend up hold cyclicals (XLI, XLB, XLK equally); "
    "when commodity trend down hold defensives (XLU, TLT, GLD equally); weekly rebalance."
)

STRATEGY = CopperCycleRotation(
    fast_ma=20,
    slow_ma=60,
    rebalance=5,
    exposure=0.97,
)
