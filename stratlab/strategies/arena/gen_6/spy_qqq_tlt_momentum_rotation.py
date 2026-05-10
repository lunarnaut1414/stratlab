"""SPY/QQQ/TLT three-way momentum rotation.

Hypothesis: Hold whichever of SPY, QQQ, or TLT has the highest 63-day
total return. Switch when the leader changes with a 3-bar minimum hold.
Full 97% concentration in the winner at all times.

Rationale:
- In bull markets: QQQ or SPY will lead
- In bond rallies / risk-off events: TLT takes over
- 63d window captures intermediate-term momentum
- 3-way rotation naturally defensive without needing explicit bear override

Structural distinctions vs existing:
- QQQ vs XLV binary uses only 2 ETFs; this uses 3 with TLT as third option
- Different from tech_vs_defensive (XLK/XLU) -- uses QQQ and SPY, not just XLK
- Different from halloween seasonality or credit-spread regime timing
- Absolute 3-way momentum (Antonacci GAA style, simplified)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ETFS = ["SPY", "QQQ", "TLT"]
MOMENTUM_WINDOW = 63
MIN_HOLD_BARS = 3
EXPOSURE = 0.97


class SpyQqqTltMomentumRotation(Strategy):
    """Top-1 of SPY/QQQ/TLT by 63d momentum; full concentration; 3-bar minimum hold."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        min_hold_bars: int = MIN_HOLD_BARS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            min_hold_bars=min_hold_bars,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.min_hold_bars = int(min_hold_bars)
        self.exposure = float(exposure)
        self._current_holding: str | None = None
        self._bars_since_switch: int = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Need enough history
        for etf in ETFS:
            try:
                hist = ctx.history(etf)
                if len(hist) < self.momentum_window + 1:
                    return []
            except Exception:
                return []

        closes = ctx.closes()
        if not all(e in closes for e in ETFS):
            return []

        prices = {e: float(closes[e]) for e in ETFS}
        if any(p <= 0 for p in prices.values()):
            return []

        # Compute momentum for each ETF
        scores: dict[str, float] = {}
        for etf in ETFS:
            try:
                hist = ctx.history(etf)
                c = hist["close"].dropna()
                if len(c) < self.momentum_window:
                    return []
                start = float(c.iloc[-self.momentum_window])
                if start <= 0:
                    continue
                ret = float(c.iloc[-1]) / start - 1.0
                if np.isfinite(ret):
                    scores[etf] = ret
            except Exception:
                return []

        if not scores:
            return []

        # Pick winner
        target_sym = max(scores, key=scores.__getitem__)
        target_price = prices[target_sym]

        # Track hold bars
        self._bars_since_switch += 1

        # Check minimum hold
        if (self._current_holding is not None
                and self._current_holding != target_sym
                and self._bars_since_switch < self.min_hold_bars):
            return []

        # If already holding the right ETF with position, nothing to do
        if self._current_holding == target_sym:
            current_pos = ctx.position(target_sym)
            if current_pos.size > 0:
                return []

        orders: list[Order] = []

        # Sell positions not in target
        for etf in ETFS:
            pos = ctx.position(etf)
            if pos.size > 0 and etf != target_sym:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=etf))

        # Estimate equity
        equity = ctx.cash
        for etf in ETFS:
            p = ctx.position(etf)
            if p.size > 0:
                equity += p.size * prices[etf]

        # Open target position
        current = int(ctx.position(target_sym).size)
        target_shares = int(equity * self.exposure / target_price)
        if target_shares > current:
            delta = target_shares - current
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))

        if orders:
            self._current_holding = target_sym
            self._bars_since_switch = 0

        return orders


UNIVERSE = ["SPY", "QQQ", "TLT"]
NAME = "spy_qqq_tlt_momentum_rotation"
HYPOTHESIS = (
    "SPY/QQQ/TLT three-way absolute momentum rotation: hold whichever of the 3 "
    "has highest 63d return; switch with 3-bar minimum hold; "
    "full 97% concentration in winner"
)
STRATEGY = SpyQqqTltMomentumRotation()
