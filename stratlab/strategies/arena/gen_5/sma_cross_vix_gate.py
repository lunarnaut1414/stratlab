"""SPY SMA crossover (10d vs 30d) with VIX calm regime gate.

Hypothesis:
  - Hold SPY when the 10-day SMA crosses above the 30-day SMA AND VIX < 25
    (calm, trending bull market condition).
  - Hold TLT when 10d SMA < 30d SMA OR VIX >= 25 (risk-off, volatile).
  - Rebalance on a weekly cadence (every 5 bars) to reduce friction.
  - Size: 97% equity exposure.

Rationale: Short-term SMA crossovers (10/30) are faster than 50/200 and
generate more trades while still filtering noise. The VIX gate prevents
holding equities during volatility spikes where correlations converge and
momentum breaks down. This combination is orthogonal to the vol-targeting
approach (spy_vol_target_trend) which holds SPY continuously in bull regimes.
"""
from __future__ import annotations

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "TLT", "^VIX"]

_SPY = "SPY"
_TLT = "TLT"
_VIX = "^VIX"
_FAST = 10
_SLOW = 30
_VIX_THRESHOLD = 25.0
_EQUITY_EXPOSURE = 0.97
_REBALANCE = 5  # weekly


class SmaVixGate(Strategy):
    """SPY 10d/30d SMA crossover gated by VIX calm regime."""

    def __init__(
        self,
        fast: int = _FAST,
        slow: int = _SLOW,
        vix_threshold: float = _VIX_THRESHOLD,
        equity_exposure: float = _EQUITY_EXPOSURE,
        rebalance: int = _REBALANCE,
    ) -> None:
        super().__init__(
            fast=fast,
            slow=slow,
            vix_threshold=vix_threshold,
            equity_exposure=equity_exposure,
            rebalance=rebalance,
        )
        self.fast = fast
        self.slow = slow
        self.vix_threshold = vix_threshold
        self.equity_exposure = equity_exposure
        self.rebalance = rebalance

    def on_bar(self, ctx: BarContext) -> list[Order]:
        min_bars = self.slow + 5
        if ctx.idx < min_bars:
            return []

        if ctx.idx % self.rebalance != 0:
            return []

        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.slow:
            return []

        vix_hist = ctx.history(_VIX)
        if len(vix_hist) < 1:
            return []

        spy_closes = spy_hist["close"]
        vix_current = float(vix_hist["close"].iloc[-1])

        sma_fast = float(spy_closes.iloc[-self.fast:].mean())
        sma_slow = float(spy_closes.iloc[-self.slow:].mean())

        bull_signal = (sma_fast > sma_slow) and (vix_current < self.vix_threshold)

        live_closes = ctx.closes()
        live_dict = {s: float(p) for s, p in live_closes.items()}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []
        target: dict[str, int] = {}

        if bull_signal:
            spy_price = live_dict.get(_SPY, 0.0)
            if spy_price > 0:
                shares = int(equity * self.equity_exposure / spy_price)
                if shares > 0:
                    target[_SPY] = shares
        else:
            tlt_price = live_dict.get(_TLT, 0.0)
            if tlt_price > 0:
                shares = int(equity * self.equity_exposure / tlt_price)
                if shares > 0:
                    target[_TLT] = shares

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        # Adjust to target
        for sym, tgt in target.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta == 0:
                continue
            if delta > 0:
                orders.append(Order(side=OrderSide.BUY, size=float(delta), symbol=sym))
            else:
                orders.append(Order(side=OrderSide.SELL, size=float(-delta), symbol=sym))

        return orders


NAME = "sma_cross_vix_gate"
HYPOTHESIS = (
    "SPY 10d vs 30d SMA crossover with VIX calm filter: hold SPY when 10d SMA > "
    "30d SMA AND VIX below 25; hold TLT when bearish; rebalance weekly; captures "
    "medium-term trend with volatility gate."
)

STRATEGY = SmaVixGate()
