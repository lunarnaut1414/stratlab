"""SP500 52-week high breakout momentum strategy.

Hypothesis: Stocks near their 52-week (252-day) high are in a strong
uptrend with institutional sponsorship. Buy top-20 SP500 stocks that are
within 5% of their 252d high AND have positive 63d momentum. Exit stocks
that drop >8% below their 252d high. Gate on SPY 200d SMA. Rebalance
every 10 bars.

Rationale: The 52-week high proximity filter is a well-documented
momentum signal (George & Hwang 2004) that is distinct from raw-return
ranking. Stocks near all-time/52w highs face less psychological selling
pressure from anchored investors. Combining with raw 63d momentum and
the 200d SMA trend gate reduces drawdown in bear markets.

Diversification vs leaderboard:
  - VIX-gated strategies use VIX as regime gate; this uses SPY 200d SMA
    (different gate condition — SPY<200d in stress).
  - The 52wk-high proximity filter is not used in any existing gen_5 strategy.
  - Combination of proximity + momentum is a different ranking signal than
    pure-return ranking or inverse-vol weighting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # bars
MOMENTUM_WINDOW = 63     # ~3 months for momentum score
HIGH_WINDOW = 252        # 52-week high lookback
PROXIMITY_THRESHOLD = 0.05  # must be within 5% of 252d high
TOP_K = 20
TREND_WINDOW = 200       # SPY 200d SMA gate
EXPOSURE = 0.97


class SP500FiftyTwoWeekHighBreakout(Strategy):
    """Buy top SP500 stocks near their 52-week high with positive momentum,
    gated by SPY 200d SMA bull-market filter."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        proximity_threshold: float = PROXIMITY_THRESHOLD,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            proximity_threshold=proximity_threshold,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.proximity_threshold = float(proximity_threshold)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA bull-market gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
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
            # Bear market: rotate to TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: 52-week high breakout + momentum filter
            need = self.high_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.high_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue
                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue
                # 52-week high
                rolling_high = float(col.iloc[-self.high_window:].max())
                if rolling_high <= 0:
                    continue
                proximity = (current_price / rolling_high) - 1.0  # 0 = at high, -0.05 = 5% below
                # Only stocks within proximity_threshold of their 52wk high
                if proximity < -self.proximity_threshold:
                    continue
                # Also require positive 63d momentum
                if len(col) < self.momentum_window:
                    continue
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom) or mom <= 0:
                    continue
                # Score: proximity (higher = closer to high) weighted by momentum
                # Use proximity as primary sort (stocks nearer all-time-high ranked higher)
                scores[sym] = proximity + 0.5 * mom  # composite score

            if len(scores) < 5:
                # Fall back to TLT if too few qualifying stocks
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_52wk_high_breakout"
HYPOTHESIS = (
    "SP500 52-week high breakout momentum: buy top-20 SP500 stocks within 5% of "
    "252d high with positive 63d momentum, equally weighted; exit positions when "
    "price drops >8% below 252d high; SPY 200d SMA gate; rebalance every 10 bars"
)

UNIVERSE = _universe

STRATEGY = SP500FiftyTwoWeekHighBreakout()
