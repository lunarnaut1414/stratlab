"""JNK 5-day return continuous credit tilt on QQQ/TLT.

Hypothesis: Replace the bilevel JNK regime allocator (used 6+ times across
the leaderboard) with a *continuous* credit tilt:

  qqq_weight = clip( 0.50 + 10 * jnk_5d_return, 0.20, 0.95 )
  tlt_weight = (1 - qqq_weight - SHY_RESERVE)

  where jnk_5d_return is JNK's 5-day total return (typically -2% to +2%).

Mapping:
  - JNK rallying +1% over 5 days -> qqq_weight = 0.60, equity tilt
  - JNK flat (0%) -> qqq_weight = 0.50, balanced
  - JNK selling -1% -> qqq_weight = 0.40, bond tilt
  - JNK selling -2% -> qqq_weight = 0.30, defensive
  - Extreme +5% rally caps at 0.95 (avoid leverage)
  - Extreme -5% sell-off floors at 0.20 (always some equity)

Why this fills a gap:
  - Phase 2 brief: "Continuous credit-spread overlay — instead of bilevel
    JNK regime, use JNK 5-day return as continuous tilt amount (not discrete
    on/off)".
  - All 6 leaderboard credit-allocator strategies use bilevel/discrete JNK
    signals (>30d MA, >20d MA, JNK/LQD ratio crossover, etc.). None compute
    a continuous tilt from JNK's recent return.
  - Differs from rp_credit_tilt which uses JNK>30d SMA as a binary "add 20%
    SPY weight" trigger; this is a smooth continuous transformation.

Universe is tiny (3 ETFs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "TLT", "JNK", "SHY"]

JNK_LOOKBACK = 5
TILT_GAIN = 10.0      # multiplier on JNK 5d return
TILT_BASE = 0.50      # neutral QQQ weight
TILT_MIN = 0.20       # min QQQ weight (heavy defensive)
TILT_MAX = 0.95       # max QQQ weight (heavy risk-on)
REBALANCE_EVERY = 5
EXPOSURE = 0.97


class JnkContinuousCreditTilt(Strategy):
    def __init__(
        self,
        jnk_lookback: int = JNK_LOOKBACK,
        tilt_gain: float = TILT_GAIN,
        tilt_base: float = TILT_BASE,
        tilt_min: float = TILT_MIN,
        tilt_max: float = TILT_MAX,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_lookback=jnk_lookback,
            tilt_gain=tilt_gain,
            tilt_base=tilt_base,
            tilt_min=tilt_min,
            tilt_max=tilt_max,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_lookback = int(jnk_lookback)
        self.tilt_gain = float(tilt_gain)
        self.tilt_base = float(tilt_base)
        self.tilt_min = float(tilt_min)
        self.tilt_max = float(tilt_max)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.jnk_lookback + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK 5d return
        jnk_ret = 0.0
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_lookback + 1:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_lookback + 1:
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_then = float(jnk_close.iloc[-self.jnk_lookback - 1])
                    if jnk_then > 0:
                        jnk_ret = jnk_now / jnk_then - 1.0
                        if not np.isfinite(jnk_ret):
                            jnk_ret = 0.0
        except KeyError:
            pass

        # Continuous QQQ weight in [tilt_min, tilt_max]
        raw = self.tilt_base + self.tilt_gain * jnk_ret
        qqq_w = max(self.tilt_min, min(self.tilt_max, raw))
        tlt_w = 1.0 - qqq_w

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        if "QQQ" in live:
            target["QQQ"] = qqq_w * self.exposure
        if "TLT" in live:
            target["TLT"] = tlt_w * self.exposure

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "jnk_continuous_credit_tilt"
HYPOTHESIS = (
    "JNK 5-day return continuous credit tilt on QQQ/TLT: qqq_weight = clip("
    "0.50 + 10*JNK_5d_return, 0.20, 0.95); TLT fills the rest; weekly "
    "rebalance; smooth continuous credit signal distinct from bilevel/MA "
    "JNK regime allocators on leaderboard."
)

STRATEGY = JnkContinuousCreditTilt()
