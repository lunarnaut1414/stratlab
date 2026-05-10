"""VIX-calm 42-day momentum on SP500 stocks.

Hypothesis:
  - When VIX < 18 (very calm regime): hold top-20 S&P 500 stocks by 42-day
    total return, equal weight.
  - When VIX >= 18: exit all positions and hold cash.
  - Rebalance every 10 trading days (bi-weekly).

Rationale: The 42-day (2-month) window lies between the well-documented
21-day (1-month) and 63-day (3-month) momentum sweet spots. Using VIX < 18
as a stricter calm-regime filter than VIX < 22 (used in vix_calm_momentum_60d)
concentrates trades in the lowest-fear environments where cross-sectional
momentum has the highest information ratio. This strategy uses a shorter
momentum window (42d vs 63d), which should differentiate it from the existing
vix_calm_momentum_60d in the correlation check.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

_VIX_THRESHOLD = 18.0
_MOMENTUM_WINDOW = 42
_TOP_K = 20
_REBALANCE = 10
_EXPOSURE = 0.97


class VixCalm42dMomentum(Strategy):
    """SP500 42-day momentum, active only when VIX < 18."""

    def __init__(
        self,
        vix_threshold: float = _VIX_THRESHOLD,
        momentum_window: int = _MOMENTUM_WINDOW,
        k: int = _TOP_K,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            vix_threshold=vix_threshold,
            momentum_window=momentum_window,
            k=k,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.vix_threshold = vix_threshold
        self.momentum_window = momentum_window
        self.k = k
        self.rebalance = rebalance
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.momentum_window + 5:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # VIX regime check
        vix_hist = ctx.history("^VIX")
        if len(vix_hist) < 1:
            return []
        current_vix = float(vix_hist["close"].iloc[-1])

        live_closes = ctx.closes()
        live_closes_dict = {s: float(p) for s, p in live_closes.items()}

        if current_vix >= self.vix_threshold:
            # Risk-off: exit everything -> cash
            orders: list[Order] = []
            for sym, pos in list(ctx.positions.items()):
                if pos.size > 0:
                    orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))
            return orders

        # Rank by momentum_window total return
        prices = ctx.closes_window(self.momentum_window + 1)
        if len(prices) < self.momentum_window:
            return []

        scores: dict[str, float] = {}
        for sym in prices.columns:
            # Skip non-tradeable symbols
            if sym.startswith("^") or sym.endswith("=F") or sym.endswith("=X"):
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window:
                continue
            ret = float(col.iloc[-1] / col.iloc[0] - 1.0)
            if not (np.isnan(ret) or np.isinf(ret)):
                scores[sym] = ret

        if len(scores) < self.k:
            return []

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        longs = set(ranked[: self.k])

        equity = ctx.portfolio_value(live_closes_dict)
        per_name = equity * self.exposure / self.k

        target: dict[str, int] = {}
        for sym in longs:
            price = live_closes_dict.get(sym)
            if price and price > 0:
                shares = int(per_name // price)
                if shares > 0:
                    target[sym] = shares

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, tgt in target.items():
            current = ctx.position(sym).size
            delta = tgt - current
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    """SP500 stocks + ^VIX for regime signal."""
    return sp500_tickers() + ["^VIX"]


NAME = "vix_calm_42d_momentum_sp500"
HYPOTHESIS = (
    "VIX-calm (VIX<18) 42-day momentum on SP500: hold top-20 stocks by "
    "42-day total return (bi-weekly rebalance) only when VIX < 18; cash otherwise. "
    "Stricter VIX gate than vix_calm_momentum_60d, shorter 42d window."
)
UNIVERSE = _universe

STRATEGY = VixCalm42dMomentum(
    vix_threshold=18.0,
    momentum_window=42,
    k=20,
    rebalance=10,
    exposure=0.97,
)
