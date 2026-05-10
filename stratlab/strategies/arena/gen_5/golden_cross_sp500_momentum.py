"""Golden-cross gated SP500 momentum.

Hypothesis: Use SPY 50d/200d SMA crossover as a macro regime signal:
  - Golden cross (50d SMA > 200d SMA): hold top-20 SP500 stocks by
    126-day (6-month) momentum, equally weighted.
  - Death cross (50d SMA <= 200d SMA): rotate to TLT 60% + SHY 40%.
  Rebalance every 10 bars.

Rationale: The golden/death cross is one of the most-watched technical
signals in equity markets. When the 50d SMA crosses above 200d, it signals
a sustained uptrend and momentum strategies perform well. This combines
macro regime timing with individual stock momentum selection.

Differentiates from gen5_bond_equity_regime (Calmar 0.64) by:
  - Uses 50d/200d SMA golden cross instead of TLT/SPY ratio MA
  - 126-day (6-month) momentum lookback vs 63-day
  - Top-20 stocks vs top-10
  - TLT+SHY 60/40 defensive (not TLT+GLD 60/40)

Differentiates from gen5_vix_gated_sp500_momentum (Calmar 0.82) by:
  - MA crossover signal (not VIX level threshold)
  - 126-day momentum (not 63-day)
  - Larger top-K (20 vs 15)
  - Different defensive assets (TLT+SHY vs SHY+TLT equal-weight)

Differentiates from gen5_sp500_trend_filter_momentum (Calmar 0.60) by:
  - 50d/200d cross vs 200d SMA price-level filter
  - 126-day momentum vs 21-day skip-1
  - 10-bar rebalance vs 21-bar
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # bars
MOMENTUM_WINDOW = 126   # ~6 months
FAST_MA = 50            # SPY fast MA
SLOW_MA = 200           # SPY slow MA
TOP_K = 20
EXPOSURE = 0.97
_BENCHMARK = "SPY"


class GoldenCrossSP500Momentum(Strategy):
    """SPY golden/death cross gated SP500 6-month momentum rotation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = rebalance_every
        self.momentum_window = momentum_window
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.top_k = top_k
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get SPY golden/death cross signal
        golden_cross = False
        try:
            spy_hist = ctx.history(_BENCHMARK)
            if spy_hist is not None and len(spy_hist) >= self.slow_ma + 2:
                spy_closes = spy_hist["close"].dropna()
                if len(spy_closes) >= self.slow_ma:
                    fast = float(spy_closes.iloc[-self.fast_ma:].mean())
                    slow = float(spy_closes.iloc[-self.slow_ma:].mean())
                    golden_cross = fast > slow
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        target: dict[str, float] = {}

        if not golden_cross:
            # Death cross: defensive TLT 60% + SHY 40%
            for sym, weight in [("TLT", 0.6), ("SHY", 0.4)]:
                if sym in closes_now.index:
                    target[sym] = weight * self.exposure
        else:
            # Golden cross: top-K momentum stocks
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                return []

            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            longs = ranked[:self.top_k]
            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SHY", "SPY"]


NAME = "golden_cross_sp500_momentum"
HYPOTHESIS = (
    "SPY 50d/200d SMA golden-cross gated SP500 momentum: hold top-20 SP500 stocks by "
    "126-day return when SPY 50d > 200d SMA; rotate to TLT 60% + SHY 40% on death cross. "
    "10-bar rebalance."
)

UNIVERSE = _universe

STRATEGY = GoldenCrossSP500Momentum()
