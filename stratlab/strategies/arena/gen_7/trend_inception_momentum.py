"""SP500 trend-inception momentum strategy.

Hypothesis: buy top-20 SP500 stocks that crossed above their 200d SMA within
last 10 bars AND have strong 21d momentum. Captures early-trend acceleration
rather than momentum continuation. Equal-weight. Biweekly rebalance.
TLT defensive when SPY below 200d SMA.

Rationale: Stocks that just crossed above their 200d SMA represent the "fresh
breakout" population — they've transitioned from downtrend to uptrend. When
combined with short-term momentum (21d return), this filters for breakouts
with price confirmation. These names are not in the near-52w-high group
(they're usually recovering from drawdowns) and not in pure continuation-
momentum groups (too early for 6-month momentum to be high). This creates a
distinct population from all existing leaderboard strategies.

Distinction from existing strategies:
  - trend-inception gate (200d SMA crossover within last N bars) is entirely
    new — no other leaderboard strategy uses it
  - Short 21d momentum window (not 42d/63d/126d used elsewhere)
  - Not inverse-vol weighted — equal weight to avoid penalizing the volatile
    breakout names that have the most upside
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 21    # short-term momentum
TREND_WINDOW = 200      # 200d SMA
INCEPTION_WINDOW = 10   # bars to look back for the SMA crossover
TOP_K = 20
EXPOSURE = 0.97


class TrendInceptionMomentum(Strategy):
    """SP500 trend-inception momentum: top-20 SP500 stocks that recently
    crossed above their 200d SMA with strong 21d momentum. TLT defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        inception_window: int = INCEPTION_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            inception_window=inception_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.inception_window = int(inception_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.inception_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA for market regime gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: defensive TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: find trend-inception stocks
            need = self.trend_window + self.inception_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.trend_window + self.inception_window:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.trend_window + self.inception_window:
                    continue

                # Compute rolling 200d SMA for the last (inception_window + 1) bars
                # We need to check if the price was below SMA some bars ago and is now above
                # The last inception_window+1 values of the series
                tail = col.iloc[-(self.trend_window + self.inception_window):]
                if len(tail) < self.trend_window + self.inception_window:
                    continue

                # For each of the last inception_window bars, compute whether price > 200d SMA
                above_flags = []
                for offset in range(self.inception_window + 1):
                    # offset=0 means the most recent bar, inception_window means oldest
                    end_idx = len(tail) - offset
                    start_idx = end_idx - self.trend_window
                    if start_idx < 0:
                        break
                    window_prices = tail.iloc[start_idx:end_idx]
                    sma_val = float(window_prices.mean())
                    current_price_val = float(tail.iloc[end_idx - 1])
                    above_flags.append(current_price_val > sma_val)

                if len(above_flags) < self.inception_window + 1:
                    continue

                # "Trend inception": currently above SMA AND was below SMA at some point
                # in the last inception_window bars (a recent crossover happened)
                currently_above = above_flags[0]  # most recent
                was_below_recently = any(not f for f in above_flags[1:])  # older bars

                if not currently_above or not was_below_recently:
                    continue  # Not a fresh crossover

                # Short-term momentum filter
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                mom = p_end / p_start - 1.0
                if not np.isfinite(mom) or mom <= 0:
                    continue  # Only positive momentum breakouts

                scores[sym] = mom

            if len(scores) < 5:
                # Not enough fresh breakouts — fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        orders: list[Order] = []

        # Liquidate positions not in target
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "trend_inception_momentum"
HYPOTHESIS = (
    "SP500 trend-inception momentum: buy top-20 SP500 stocks that crossed above "
    "their 200d SMA within last 10 bars AND have strong 21d momentum; captures "
    "early-trend acceleration rather than momentum continuation; equal-weight; "
    "biweekly rebalance; TLT defensive when SPY below 200d SMA"
)

UNIVERSE = _universe

STRATEGY = TrendInceptionMomentum()
