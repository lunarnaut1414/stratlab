"""Equity risk premium harvesting via trend-filtered SP500 momentum.

Hypothesis: Hold top-20 SP500 stocks by 1-month (21-day) momentum
(skipping the most recent 1 day to avoid reversal) when SPY is above its
200-day SMA. In downtrend (SPY < 200d SMA), hold all capital in TLT.
Rebalance every 21 trading bars (~monthly cadence).

Key design differences from gen5_bond_equity_regime (Calmar 0.64):
  - Uses SPY > 200d SMA as trend gate (not TLT/SPY ratio MA crossover)
  - Uses 21-day skip-1-day momentum (short-term momentum with reversal skip)
  - Top-20 stocks for broader coverage
  - Monthly rebalance (every 21 bars)
  - Pure TLT in defensive mode (not TLT+GLD mix)

Rationale: SPY 200d SMA is a cleaner binary signal than the TLT/SPY ratio
which can whipsaw. The short-term 21-day return with 1-day skip avoids the
known 1-month reversal effect in individual stocks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21   # bars (~monthly)
MOMENTUM_WINDOW = 21   # ~1 month
SKIP_DAYS = 1          # skip most recent day to avoid reversal
TREND_WINDOW = 200     # SPY 200d SMA
TOP_K = 20
EXPOSURE = 0.97
_BENCHMARK = "SPY"


class SP500TrendMomentum(Strategy):
    """SPY-200d-SMA gated momentum: top-20 SP500 by 1-month momentum in uptrend, else TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        skip_days: int = SKIP_DAYS,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            skip_days=skip_days,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = rebalance_every
        self.momentum_window = momentum_window
        self.skip_days = skip_days
        self.trend_window = trend_window
        self.top_k = top_k
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window + self.skip_days) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Determine SPY trend
        spy_in_uptrend = False
        try:
            spy_hist = ctx.history(_BENCHMARK)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_closes = spy_hist["close"].dropna()
                if len(spy_closes) >= self.trend_window:
                    spy_sma = float(spy_closes.iloc[-self.trend_window:].mean())
                    spy_now = float(spy_closes.iloc[-1])
                    spy_in_uptrend = spy_now > spy_sma
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

        if not spy_in_uptrend:
            # Defensive: all in TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K momentum stocks (skip last 1 day to avoid reversal)
            lookback_needed = self.momentum_window + self.skip_days + 5
            prices = ctx.closes_window(lookback_needed)
            if len(prices) < self.momentum_window + self.skip_days:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.skip_days:
                    continue
                # 21-day return ending skip_days ago
                end_idx = -(self.skip_days)    # exclude most recent skip_days
                start_idx = -(self.momentum_window + self.skip_days)
                price_end = float(col.iloc[end_idx])
                price_start = float(col.iloc[start_idx])
                if price_start > 0 and np.isfinite(price_end) and np.isfinite(price_start):
                    ret = (price_end - price_start) / price_start
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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_trend_filter_momentum"
HYPOTHESIS = (
    "SPY-200d-SMA gated SP500 momentum: hold top-20 SP500 stocks by 21-day return "
    "(skip 1 day to avoid reversal) when SPY above 200d SMA; hold TLT when below. "
    "Monthly rebalance (every 21 bars)."
)

UNIVERSE = _universe

STRATEGY = SP500TrendMomentum()
