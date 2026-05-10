"""JNK credit + VIX dual-gate QQQ/SPY/TLT three-tier strategy.

Hypothesis: Combine credit spread trend (JNK vs its 20d MA) and VIX level
to create a 3-tier allocation:
  - Tier 1 (high confidence bull): JNK > 20d MA AND VIX < 20 → QQQ 97%
  - Tier 2 (moderate bull): JNK > 20d MA AND VIX 20-28 → SPY 97%
  - Tier 3 (risk-off): JNK < 20d MA → SHY 50% + TLT 47%

Rationale: The credit signal (JNK trend) tells us whether the credit cycle
is supporting equities. VIX level then determines whether to take tech-heavy
(QQQ) or broad-market (SPY) exposure. This dual-signal approach differs from:
- `hy_credit_qqq_rotation`: uses JNK>30d MA AND SPY>100d MA (equity trend, not VIX)
- `jnk_lqd_spy_regime`: uses JNK/LQD ratio (relative credit), not VIX
- `sma_cross_vix_gate`: uses SPY SMA crossover + VIX (no credit signal)

The combination of credit + VIX creates different timing from either alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["JNK", "QQQ", "SPY", "TLT", "SHY", "^VIX"]

JNK_MA = 20           # JNK MA period for credit trend
VIX_CALM_THRESHOLD = 20.0    # below this: use QQQ
VIX_CAUTION_THRESHOLD = 28.0  # above this: risk-off
REBALANCE_EVERY = 5   # weekly
EXPOSURE = 0.97


class JnkVixDualGateQqq(Strategy):
    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        vix_calm: float = VIX_CALM_THRESHOLD,
        vix_caution: float = VIX_CAUTION_THRESHOLD,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            vix_calm=vix_calm,
            vix_caution=vix_caution,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.vix_calm = float(vix_calm)
        self.vix_caution = float(vix_caution)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.jnk_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK credit signal
        credit_bullish = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma + 1:
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    credit_bullish = jnk_now > jnk_ma_val
        except KeyError:
            pass

        # VIX level
        vix_level = 20.0  # neutral default
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= 2:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= 1:
                    vix_level = float(vix_close.iloc[-1])
        except KeyError:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # 3-tier allocation
        target: dict[str, float] = {}

        if not credit_bullish:
            # Risk-off: credit market is deteriorating
            if "SHY" in live:
                target["SHY"] = 0.50 * self.exposure
            if "TLT" in live:
                target["TLT"] = 0.50 * self.exposure
            if not target:
                if "TLT" in live:
                    target["TLT"] = self.exposure
        elif vix_level < self.vix_calm:
            # Tier 1: Credit good + low VIX → aggressive QQQ
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        elif vix_level < self.vix_caution:
            # Tier 2: Credit good + moderate VIX → SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
        else:
            # Tier 3: Credit good but VIX very elevated → cautious SPY
            # (VIX > 28 but credit still OK - stay defensive with SPY)
            if "SPY" in live:
                target["SPY"] = 0.60 * self.exposure
            if "TLT" in live:
                target["TLT"] = 0.40 * self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
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


NAME = "jnk_vix_dual_gate_qqq"
HYPOTHESIS = (
    "JNK credit + VIX dual-gate QQQ/SPY/TLT 3-tier: credit strong (JNK>20d MA) "
    "AND VIX calm (VIX<20): QQQ 97%; credit strong but VIX elevated (20-28): SPY 97%; "
    "credit weak (JNK<20d MA): SHY 50%+TLT 47%; weekly rebalance; dual credit+vol "
    "signal creates different timing from single-signal variants"
)

STRATEGY = JnkVixDualGateQqq()
