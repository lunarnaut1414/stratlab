"""JNK credit momentum + SPY breadth → QQQ rotation — gen_6 sonnet-7

Hypothesis: Hold QQQ 97% when JNK 10d MA > 40d MA (credit momentum positive)
AND SPY is above 100d SMA. Hold SPY 97% when SPY is bullish but credit
momentum has turned negative. Hold TLT 97% when SPY is below 100d SMA
(bear market). Rebalance every 5 bars.

This is a simplified 3-state version that:
  1. Uses JNK 10d/40d MA crossover (shorter MAs than JNK 20d/60d used elsewhere)
  2. Routes credit-healthy state to QQQ (not SP500 stocks)
  3. Routes credit-weak-but-bull state to SPY (not SHY/TLT)
  4. Routes bear market to TLT only

Distinct from gen6_jnk_lqd_spy_regime (which uses JNK 20d/60d MA + SSO leverage)
and gen6_hy_credit_qqq_rotation (which uses JNK 30d SMA + SPY 100d SMA → QQQ only).
Different MA windows (10d/40d) create different signal timing.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5   # weekly
JNK_FAST = 10         # JNK fast MA
JNK_SLOW = 40         # JNK slow MA (different from 20/60 or 30d SMA used elsewhere)
TREND_WINDOW = 100    # SPY 100d SMA
EXPOSURE = 0.97


class JNKSPYBreadthQQQ(Strategy):
    """JNK 10/40 crossover + SPY 100d SMA → QQQ/SPY/TLT 3-state."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_fast: int = JNK_FAST,
        jnk_slow: int = JNK_SLOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_fast=jnk_fast,
            jnk_slow=jnk_slow,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_fast = int(jnk_fast)
        self.jnk_slow = int(jnk_slow)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_slow, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
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
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # JNK credit momentum (10d/40d MA crossover)
        jnk_bullish = True  # assume bullish if no data
        try:
            jnk_hist = ctx.history("JNK")
            if len(jnk_hist) >= self.jnk_slow + 2:
                jnk_close = jnk_hist["close"].dropna().values
                jnk_fast_ma = float(np.mean(jnk_close[-self.jnk_fast:]))
                jnk_slow_ma = float(np.mean(jnk_close[-self.jnk_slow:]))
                jnk_bullish = jnk_fast_ma > jnk_slow_ma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif jnk_bullish:
            # Credit momentum positive + bull: QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # Credit momentum negative but SPY in uptrend: SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


NAME = "jnk_spy_breadth_qqq"
HYPOTHESIS = (
    "JNK 10d/40d MA crossover + SPY 100d SMA → QQQ/SPY/TLT: JNK bullish AND SPY bull → QQQ 97%; "
    "JNK bearish but SPY bull → SPY 97%; SPY bear → TLT 97%; weekly rebalance; "
    "shorter JNK MA windows (10/40) create different timing than existing 20/60 or 30d variants"
)
UNIVERSE = ["QQQ", "SPY", "TLT", "JNK"]
STRATEGY = JNKSPYBreadthQQQ()
