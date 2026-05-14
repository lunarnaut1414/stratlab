"""SP500 Momentum with JNK Credit Pulse Gate — gen_8 sonnet-10

Hypothesis: Hold top-15 SP500 stocks by 63d return when BOTH:
  1. SPY is above its 200d SMA (bull market regime)
  2. JNK's 5-day return is positive (credit expanding/healthy)

When SPY is bull but JNK's 5-day credit pulse is negative (credit contracting),
hold SPY 97% instead of concentrated stock selection — the broad market can still
perform but individual stock concentration is too risky.

When SPY is below 200d SMA (bear), rotate fully to TLT.

Rationale: Most leaderboard JNK-gated strategies use JNK vs a 20d or 30d MA
(a slow, lagging signal). The 5-day JNK return is a fine-grained credit momentum
pulse — it detects credit stress or expansion at a weekly timescale, much faster
than MA-crossover signals. When credit is expanding, stock selection momentum
works well. When credit contracts, even good momentum stocks get dragged down.

Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # ~3 months for stock selection
SPY_TREND_WINDOW = 200    # SPY bear gate
JNK_PULSE_WINDOW = 5      # 5-day JNK return as credit pulse
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_JNK = "JNK"


class SP500JNKCreditPulse(Strategy):
    """SP500 momentum gated by SPY trend + JNK 5-day credit pulse."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        jnk_pulse_window: int = JNK_PULSE_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            jnk_pulse_window=jnk_pulse_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.jnk_pulse_window = int(jnk_pulse_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY bear gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # JNK 5-day credit pulse
        jnk_expanding = True  # default if signal unavailable
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_pulse_window + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_pulse_window + 1:
                    jnk_ret_5d = float(
                        jnk_close.iloc[-1] / jnk_close.iloc[-self.jnk_pulse_window - 1] - 1.0
                    )
                    jnk_expanding = jnk_ret_5d > 0
        except (KeyError, Exception):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: full TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        elif not jnk_expanding:
            # Bull market but credit contracting: hold broad SPY (reduced risk)
            if _SPY in live:
                target[_SPY] = self.exposure

        else:
            # Bull + credit expanding: top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _JNK):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
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
    return sp500_tickers() + [_TLT, _SPY, _JNK]


NAME = "sp500_jnk_credit_pulse"
HYPOTHESIS = (
    "SP500 momentum with JNK credit acceleration entry gate: hold top-15 SP500 stocks by "
    "63d return when SPY>200d SMA AND JNK 5d return > 0 (credit expanding); when JNK 5d "
    "return negative (credit contracting) hold SPY 97% instead of stock selection; when "
    "SPY<200d SMA hold TLT; biweekly rebalance; JNK 5-day return as fine-grained credit "
    "pulse distinct from JNK MA-level gates on leaderboard"
)

UNIVERSE = _universe

STRATEGY = SP500JNKCreditPulse()
