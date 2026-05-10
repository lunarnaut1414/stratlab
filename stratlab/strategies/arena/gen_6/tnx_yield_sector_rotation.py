"""TNX yield-trend sector rotation strategy.

Hypothesis: The direction of the 10-year Treasury yield (^TNX) drives
cross-sector returns in a systematic way. Financial and energy sectors
outperform when rates are rising; utilities and long-duration bonds
outperform when rates are falling.

Signal: Compare 20d MA vs 60d MA of ^TNX
  - Rising rates (20d MA >= 60d MA): hold XLF 50% + XLE 47%
    (financials benefit from steeper NIM; energy benefits from inflation)
  - Falling rates (20d MA < 60d MA): hold XLU 50% + TLT 47%
    (utilities bond-proxy; TLT benefits from yield decline)
  - Rebalance weekly (every 5 bars)

Rationale: Rate-sensitive sectors have well-documented factor tilts.
Financials (XLF) benefit from rate rise through net interest margin expansion.
Utilities (XLU) are rate-sensitive bond proxies that underperform as rates
rise. Energy (XLE) tends to correlate with inflation, which often accompanies
rising rates. TLT is the clearest beneficiary of rate declines.

This signal is orthogonal to all gen_5 strategies:
  - No VIX gating, no credit spread, no momentum ranking
  - Pure yield-direction as primary signal (^TNX is signal-only; route
    exposure through ETFs XLF, XLE, XLU, TLT)
  - rsi_etf_meanrev used TNX-IRX spread level (yield CURVE shape), this
    uses TNX direction (rate TREND) — different approach
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 20      # 20d MA of TNX
SLOW_MA = 60      # 60d MA of TNX
REBALANCE = 5     # weekly
EXPOSURE = 0.97
_TNX = "^TNX"


class TNXYieldSectorRotation(Strategy):
    """10yr-yield trend drives sector rotation: rising rates -> XLF+XLE; falling -> XLU+TLT."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Read ^TNX (10-year treasury yield) — signal-only, not tradeable
        try:
            tnx_hist = ctx.history(_TNX)
        except KeyError:
            return []
        if tnx_hist is None or len(tnx_hist) < self.slow_ma + 2:
            return []

        tnx_close = tnx_hist["close"].dropna()
        if len(tnx_close) < self.slow_ma:
            return []

        fast_val = float(tnx_close.iloc[-self.fast_ma:].mean())
        slow_val = float(tnx_close.iloc[-self.slow_ma:].mean())
        rates_rising = fast_val >= slow_val

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target allocation
        if rates_rising:
            # Rising rates: financials + energy outperform
            allocations = [("XLF", 0.50 * self.exposure), ("XLE", 0.47 * self.exposure)]
        else:
            # Falling rates: utilities + long bonds outperform
            allocations = [("XLU", 0.50 * self.exposure), ("TLT", 0.47 * self.exposure)]

        target: dict[str, float] = {}
        for sym, weight in allocations:
            if sym in closes_now.index:
                target[sym] = weight

        # Fallback to SPY if neither target ETF available
        if not target:
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "tnx_yield_sector_rotation"
HYPOTHESIS = (
    "TNX yield-trend sector rotation: use 10yr treasury yield (^TNX) 20d vs 60d MA crossover; "
    "rising rates hold XLF 50%+XLE 47%; falling rates hold XLU 50%+TLT 47%; "
    "captures rate-driven sector factor rotation orthogonal to existing leaderboard; weekly rebalance"
)

UNIVERSE = ["XLF", "XLE", "XLU", "TLT", "SPY", _TNX]

STRATEGY = TNXYieldSectorRotation()
