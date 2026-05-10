"""SP500 price-momentum acceleration strategy — gen_6 sonnet-4

Hypothesis: Stocks where short-term (10d) momentum exceeds long-term (40d)
momentum show an acceleration in price trend — a sign of strengthening
momentum. This is distinct from pure momentum (which ranks by single
lookback) and captures the "rate of change" in momentum.

Gate: SPY above 200d SMA (bull market only). Rotate to TLT when bearish.
Biweekly rebalance (every 10 bars).

Distinct from existing leaderboard:
  - Not VIX-gated (uses SPY trend only)
  - Acceleration signal (ratio of short/long momentum) not raw momentum
  - No skip-period, no inverse-vol weighting
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10   # biweekly
SHORT_WINDOW = 10      # short-term momentum
LONG_WINDOW = 40       # long-term momentum
TREND_WINDOW = 200     # SPY bull/bear gate
TOP_K = 20
EXPOSURE = 0.97


class SP500MomentumAcceleration(Strategy):
    """Top SP500 stocks by momentum acceleration (10d > 40d), SPY 200d gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_window=short_window,
            long_window=long_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.long_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: full TLT allocation
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K by momentum acceleration
            need = self.long_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.long_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.long_window + 2:
                    continue
                # Short-term return (last 10d)
                p_now = float(col.iloc[-1])
                p_short = float(col.iloc[-self.short_window - 1])
                p_long = float(col.iloc[-self.long_window - 1])
                if p_short <= 0 or p_long <= 0:
                    continue
                ret_short = p_now / p_short - 1.0
                ret_long = p_now / p_long - 1.0
                if not np.isfinite(ret_short) or not np.isfinite(ret_long):
                    continue
                # Acceleration = short-term minus long-term return
                # A positive value means recent trend is stronger than baseline
                accel = ret_short - ret_long
                # Also require positive long-term momentum (avoid catching falling knives)
                if ret_long > 0:
                    scores[sym] = accel

            if len(scores) < self.top_k:
                return []

            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:self.top_k]
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


NAME = "sp500_momentum_acceleration"
HYPOTHESIS = (
    "SP500 price-momentum acceleration: buy top-20 stocks where 10d return > 40d return "
    "(short-term beat long-term, momentum strengthening), above SPY 200d SMA; "
    "equal-weight; biweekly rebalance; TLT defensive"
)

UNIVERSE = _universe

STRATEGY = SP500MomentumAcceleration()
