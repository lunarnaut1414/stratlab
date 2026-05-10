"""TNX yield-trend sector rotation strategy.

Hypothesis:
  Use the 10-year Treasury yield (^TNX) 20d vs 60d MA crossover to rotate
  between rate-sensitive sectors:
    - Rising rates (TNX 20d MA > 60d MA): hold XLF 50% + XLE 47% (financials
      and energy benefit from higher rates and reflation)
    - Falling rates (TNX 20d MA < 60d MA): hold XLU 50% + TLT 47% (utilities
      and long bonds benefit from falling rates)
    - Neutral / no signal: hold SPY 97% (equity default)
  Neutral is defined as: within 0.05 bps band to prevent whipsawing.

Rationale:
  Interest rate regime fundamentally drives sector leadership. XLF benefits
  from steeper yield curves and wider net interest margins. XLE is correlated
  with inflation/reflation which also lifts rates. XLU is a bond proxy that
  outperforms in falling-rate regimes. This is orthogonal to VIX, credit
  spread, breadth, and price-momentum signals used by existing leaderboard.

  Key distinctions from existing strategies:
  - Signal is TNX yield direction, not VIX level or SMA crossover on equity
  - Sector allocation (XLF/XLE vs XLU) not factor or cross-asset allocation
  - Weekly rebalance (5 bars)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

TNX = "^TNX"
FAST_MA = 20
SLOW_MA = 60
REBALANCE = 5   # weekly
EXPOSURE = 0.97
NEUTRAL_BAND = 0.05   # % difference within which we go to SPY

RISING_ALLOCATION = [("XLF", 0.50), ("XLE", 0.47)]
FALLING_ALLOCATION = [("XLU", 0.50), ("TLT", 0.47)]
NEUTRAL_ALLOCATION = [("SPY", 0.97)]


class TNXRateSectorRotation(Strategy):
    """Rate-regime sector rotation: XLF+XLE rising, XLU+TLT falling, SPY neutral."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
        neutral_band: float = NEUTRAL_BAND,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
            neutral_band=neutral_band,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)
        self.neutral_band = float(neutral_band)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Get TNX history for yield-trend signal
        try:
            tnx_hist = ctx.history(TNX)
        except KeyError:
            return []
        if tnx_hist is None or len(tnx_hist) < self.slow_ma + 5:
            return []

        tnx_close = tnx_hist["close"].dropna()
        if len(tnx_close) < self.slow_ma:
            return []

        fast = float(tnx_close.iloc[-self.fast_ma:].mean())
        slow = float(tnx_close.iloc[-self.slow_ma:].mean())
        if slow <= 0:
            return []

        spread_pct = (fast - slow) / abs(slow)

        # Determine regime
        if spread_pct > self.neutral_band:
            allocation = RISING_ALLOCATION
        elif spread_pct < -self.neutral_band:
            allocation = FALLING_ALLOCATION
        else:
            allocation = NEUTRAL_ALLOCATION

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Build target
        target: dict[str, float] = {}
        for sym, w in allocation:
            if sym in closes_now.index:
                target[sym] = w

        if not target:
            # Fallback to SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


NAME = "tnx_rate_sector_rotation"
HYPOTHESIS = (
    "TNX yield-trend sector rotation: use 10yr Treasury yield (^TNX) 20d vs 60d MA crossover; "
    "rising rates hold XLF 50%+XLE 47%; falling rates hold XLU 50%+TLT 47%; "
    "neutral hold SPY 97%; captures rate-driven sector factor rotation orthogonal to "
    "existing leaderboard; weekly rebalance with 5bps neutral band."
)

UNIVERSE = ["XLF", "XLE", "XLU", "TLT", "SPY", TNX]

STRATEGY = TNXRateSectorRotation()
