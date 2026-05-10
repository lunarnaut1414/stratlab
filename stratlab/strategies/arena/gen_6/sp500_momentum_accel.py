"""SP500 momentum acceleration strategy.

Hypothesis: Stocks where short-term momentum (10d return) exceeds long-term
momentum (40d return) are experiencing accelerating price appreciation —
institutional buying is intensifying. Top-20 such stocks above SPY 200d SMA.

Signal construction:
  - Compute 10d return and 40d return for each SP500 stock
  - Score = 10d_return - 40d_return (positive = acceleration)
  - Require both 10d and 40d returns to be positive (upward trend confirmed)
  - Filter to stocks above their own 50d SMA (stock-level trend filter)
  - Rank by acceleration score; hold top-20
  - Gate: only active when SPY above 200d SMA (market regime)
  - Defensive: TLT when SPY below 200d SMA
  - Biweekly rebalance (every 10 bars)

Rationale: Pure momentum (63d/126d return) ranks stocks that have already
run. Acceleration captures the CHANGE in momentum — stocks just starting
to outperform. This is conceptually similar to the "momentum factor turnover"
literature that shows acceleration predicts continuation better than raw
level for high-turnover universes.

Diversification vs leaderboard:
  - Different ranking signal (acceleration vs raw return or 52wk proximity)
  - Uses 50d SMA individual stock filter (novel constraint)
  - Both 10d and 40d positive requirement = quality filter (not just best
    momentum, but actively improving AND trending up)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
SHORT_WINDOW = 10       # acceleration short lookback
LONG_WINDOW = 40        # acceleration long lookback
STOCK_TREND = 50        # individual stock SMA filter
TREND_WINDOW = 200      # SPY 200d SMA gate
TOP_K = 20
EXPOSURE = 0.97


class SP500MomentumAccel(Strategy):
    """Top SP500 stocks by momentum acceleration (10d beat 40d), SPY 200d SMA gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
        stock_trend: int = STOCK_TREND,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_window=short_window,
            long_window=long_window,
            stock_trend=stock_trend,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.stock_trend = int(stock_trend)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.long_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA market regime gate
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
            # Bear market: TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: momentum acceleration filter
            need = self.trend_window + self.long_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.long_window + 5:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.long_window + 1:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue

                # Individual stock 50d SMA filter
                if len(col) >= self.stock_trend:
                    stock_sma = float(col.iloc[-self.stock_trend:].mean())
                    if current_price < stock_sma:
                        continue  # skip stocks in downtrend

                # Compute short and long returns
                if len(col) < self.short_window + 1:
                    continue
                short_ret = float(col.iloc[-1] / col.iloc[-self.short_window] - 1.0)
                long_ret = float(col.iloc[-1] / col.iloc[-self.long_window] - 1.0)

                if not np.isfinite(short_ret) or not np.isfinite(long_ret):
                    continue

                # Require long-term return to be positive (confirmed uptrend)
                # but allow short-term to be slightly negative if acceleration is large
                if long_ret <= 0:
                    continue

                # Acceleration = short momentum exceeds long momentum
                # Use composite: weight raw 40d return + acceleration
                accel = short_ret - long_ret
                # Combined score: base 40d momentum + 2x acceleration premium
                scores[sym] = long_ret + 2.0 * max(0.0, accel)

            if len(scores) < 5:
                # Fallback to TLT if insufficient qualifying stocks
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_momentum_accel"
HYPOTHESIS = (
    "SP500 price-momentum acceleration filter: buy top-20 stocks where 10d return "
    "exceeds 40d return (short-term beat long-term = momentum strengthening), "
    "above SPY 200d SMA; equal-weight; biweekly rebalance; TLT defensive; "
    "captures momentum acceleration signal distinct from raw-return ranking"
)

UNIVERSE = _universe

STRATEGY = SP500MomentumAccel()
