"""SP500 momentum with within-stock trend confirmation and JNK credit gate.

Hypothesis: hold top-20 SP500 stocks by 63d return where the stock's own
20d SMA > 60d SMA (trend-confirmed momentum); JNK above 30d SMA AND SPY
above 200d SMA; equal-weight; rebalance every 10 bars; TLT defensive.

Rationale: Pure cross-sectional momentum captures recent outperformers but
includes stocks that spiked then are now fading (e.g., short squeezes). Adding
a within-stock trend confirmation (20d SMA > 60d SMA) filters out "dying
momentum" stocks and keeps only those in active uptrends. This is different from:
  - near-52w-high filter (uses price relative to annual high)
  - RSI confirmation (uses oscillator)
  - Sharpe filter (uses return/vol)

The JNK credit gate and SPY 200d SMA gate together ensure this only runs
in healthy market environments where momentum factor performs.

Distinction from existing strategies:
  - Within-stock 20d/60d SMA golden cross as quality/trend confirmation filter
  - JNK 30d SMA credit gate + SPY 200d SMA (dual gate)
  - Top-20 equal-weight (different from inverse-vol weighted nearhi)
  - Biweekly rebalance generates high trade count
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # ~3 months
FAST_MA = 20               # within-stock fast SMA
SLOW_MA = 60               # within-stock slow SMA
JNK_MA = 30               # JNK SMA for credit regime
SPY_TREND_WINDOW = 200     # SPY SMA for market trend
TOP_K = 20
EXPOSURE = 0.97


class ConfirmedMomentumJnkGated(Strategy):
    """Top-20 SP500 by 63d momentum with 20d/60d SMA within-stock confirmation; JNK + SPY gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        jnk_ma: int = JNK_MA,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            jnk_ma=jnk_ma,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.jnk_ma = int(jnk_ma)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.slow_ma, self.spy_trend_window, self.jnk_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # Check JNK credit regime
        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return []
        if len(jnk_hist) < self.jnk_ma + 2:
            return []
        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) < self.jnk_ma:
            return []
        jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
        jnk_risk_on = float(jnk_close.iloc[-1]) > jnk_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        risk_on = spy_bull and jnk_risk_on

        if not risk_on:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K momentum stocks with within-stock SMA confirmation
            need = max(self.momentum_window, self.slow_ma) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < need - 3:
                    continue

                # Momentum score (63d return)
                if len(col) < self.momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Within-stock SMA confirmation: 20d SMA > 60d SMA
                if len(col) < self.slow_ma:
                    continue
                fast_sma = float(col.iloc[-self.fast_ma:].mean())
                slow_sma = float(col.iloc[-self.slow_ma:].mean())
                if fast_sma <= slow_sma:
                    continue  # Skip stocks in a downtrend

                scores[sym] = ret

            if len(scores) < 5:
                # Not enough confirmed stocks — go defensive
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / k
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
    return sp500_tickers() + ["TLT", "SPY", "JNK"]


NAME = "confirmed_momentum_jnk_gated"
HYPOTHESIS = (
    "SP500 momentum with within-stock 20d/60d SMA confirmation and JNK credit gate: hold top-20 "
    "SP500 stocks by 63d return where the stock's own 20d SMA > 60d SMA (trend-confirmed momentum); "
    "JNK above 30d SMA AND SPY above 200d SMA; equal-weight; rebalance every 10 bars; TLT defensive"
)

UNIVERSE = _universe

STRATEGY = ConfirmedMomentumJnkGated()
