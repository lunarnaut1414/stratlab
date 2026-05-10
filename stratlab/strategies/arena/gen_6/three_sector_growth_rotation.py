"""Three-sector growth rotation strategy.

Hypothesis: Rank XLK (tech), XLV (healthcare), and XLI (industrials) by
60-day return. Hold the top-1 sector ETF fully invested (97%). Switch when
the leader changes. 5-day minimum hold to avoid rapid switching.
SPY 200d SMA bear override: rotate to IEF (mid-duration treasury).

Rationale:
- Tech (XLK): growth leader in expansions
- Healthcare (XLV): defensive growth, outperforms in slowdowns
- Industrials (XLI): cyclical leader, signals peak/early recovery

These three sectors capture distinct phases of the economic cycle.
Rotating to the current leader captures momentum within quality sectors.

Structural distinctions vs leaderboard:
- Only 3 sector ETFs considered (not broad SP500 stock selection)
- Top-1 concentrated holding (no diversification within equities)
- 5-day minimum hold prevents daily churn
- Different from tech_vs_defensive (XLK/XLU): uses XLV/XLI instead of XLU
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTORS = ["XLK", "XLV", "XLI"]
MOMENTUM_WINDOW = 60
MIN_HOLD_BARS = 5
TREND_WINDOW = 200
REBALANCE_EVERY = 5
EXPOSURE = 0.97
DEFENSIVE = "IEF"


class ThreeSectorGrowthRotation(Strategy):
    """Top-1 of XLK/XLV/XLI by 60d momentum with SPY 200d bear override."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        min_hold_bars: int = MIN_HOLD_BARS,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            min_hold_bars=min_hold_bars,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.min_hold_bars = int(min_hold_bars)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self._current_leader: str | None = None
        self._bars_since_switch: int = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []

        # Track bars since last switch
        self._bars_since_switch += 1

        # Only consider switching every rebalance_every bars
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY trend gate
        bull = True
        try:
            spy_hist = ctx.history("SPY")
            spy_c = spy_hist["close"].dropna()
            if len(spy_c) >= self.trend_window:
                bull = float(spy_c.iloc[-1]) > float(spy_c.iloc[-self.trend_window:].mean())
        except Exception:
            pass

        new_leader: str
        if not bull:
            new_leader = DEFENSIVE
        else:
            # Rank sectors by momentum
            scores: dict[str, float] = {}
            for sym in SECTORS:
                try:
                    hist = ctx.history(sym)
                    c = hist["close"].dropna()
                    if len(c) >= self.momentum_window:
                        ret = float(c.iloc[-1] / c.iloc[-self.momentum_window] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret
                except Exception:
                    pass

            if not scores:
                return []

            new_leader = max(scores, key=scores.__getitem__)

        # Minimum hold check: don't switch if too soon
        if (self._current_leader is not None
                and new_leader != self._current_leader
                and self._bars_since_switch < self.min_hold_bars):
            return []

        # If leader unchanged, no trades needed
        if new_leader == self._current_leader:
            return []

        # Switch to new leader
        self._current_leader = new_leader
        self._bars_since_switch = 0

        target: dict[str, float] = {new_leader: self.exposure}

        orders: list[Order] = []

        # Sell positions not in target
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


UNIVERSE = ["XLK", "XLV", "XLI", "IEF", "SPY"]
NAME = "three_sector_growth_rotation"
HYPOTHESIS = (
    "Three-sector growth rotation: rank XLK/XLV/XLI by 60d return; "
    "hold top-1 sector ETF fully invested; switch when leader changes "
    "with 5-day minimum hold; SPY 200d SMA bear override to IEF; rebalance weekly"
)
STRATEGY = ThreeSectorGrowthRotation()
