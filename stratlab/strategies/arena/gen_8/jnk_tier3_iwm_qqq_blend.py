"""JNK 3-Tier Credit Gating IWM+QQQ Blend — gen_8 sonnet-4

Hypothesis: Use JNK as a 3-tier credit regime signal:
  Tier 1 (strong credit): JNK above 50d SMA AND JNK 10d return > 0
    → hold IWM 48% + QQQ 49% (risk-on: small-cap + growth blend)
  Tier 2 (mild credit): JNK above 50d SMA but 10d return <= 0
    → hold SPY 97% (moderate risk-on: stable broad market)
  Tier 3 (credit stress): JNK below 50d SMA
    → hold TLT 97% (defensive)

Weekly rebalance.

Rationale: JNK with 2 dimensions (level vs MA, and short-term momentum) creates
3 distinct credit states:
  - "Strong" credit (above MA + recent momentum): risk appetite expanding → aggressive
  - "Stable" credit (above MA, momentum stalled): neutral → broad market
  - "Weak" credit (below MA): risk aversion → defensive

In 2010-2018 IS window, JNK was above its 50d MA for ~75% of the period, with
strong short-term momentum for ~50%. IWM+QQQ blend outperforms SPY alone in
the strong-credit regime (small-cap and growth both benefit from easy credit).

Distinction from existing strategies:
  - gen6_jnk_vix_dual_gate_qqq: uses VIX as second gate (not JNK short-term momentum)
    and routes to QQQ only (not IWM+QQQ blend), with only 2 equity tiers (QQQ vs SPY)
  - gen6_hy_credit_qqq_rotation: JNK+SPY 100d SMA → QQQ or TLT (only 2 states)
  - gen6_jnk_continuous_credit_tilt: continuous weighting, QQQ/TLT only (no IWM)
  - This is 3-discrete-tier with IWM inclusion in the aggressive leg — novel combination
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
JNK_MA = 50                # JNK slow trend MA
JNK_SHORT = 10             # JNK short-term momentum window
EXPOSURE = 0.97


class JnkTier3IwmQqqBlend(Strategy):
    """3-tier JNK credit gating: IWM+QQQ blend (strong) / SPY (mild) / TLT (stress)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        jnk_short: int = JNK_SHORT,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            jnk_short=jnk_short,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.jnk_short = int(jnk_short)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.jnk_short) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # JNK signals
        jnk_above_ma = False
        jnk_short_positive = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma + 2:
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_above_ma = jnk_now > jnk_ma_val

                    if len(jnk_close) >= self.jnk_short + 1:
                        jnk_past = float(jnk_close.iloc[-self.jnk_short])
                        if jnk_past > 0:
                            jnk_short_ret = jnk_now / jnk_past - 1.0
                            jnk_short_positive = np.isfinite(jnk_short_ret) and jnk_short_ret > 0
        except Exception:
            pass

        target: dict[str, float] = {}

        if jnk_above_ma and jnk_short_positive:
            # Tier 1: Strong credit — IWM 48% + QQQ 49%
            if "IWM" in live:
                target["IWM"] = 0.48 * self.exposure
            if "QQQ" in live:
                target["QQQ"] = 0.49 * self.exposure
        elif jnk_above_ma:
            # Tier 2: Mild credit — SPY 97%
            if "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Tier 3: Credit stress — TLT 97%
            if "TLT" in live:
                target["TLT"] = self.exposure

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


UNIVERSE = ["QQQ", "IWM", "SPY", "TLT", "JNK"]

NAME = "jnk_tier3_iwm_qqq_blend"
HYPOTHESIS = (
    "JNK 3-tier credit gating IWM+QQQ blend vs SPY vs TLT: when JNK above 50d SMA AND "
    "JNK 10d return positive (strong credit momentum) hold IWM 48%+QQQ 49% (risk-on "
    "growth+smallcap blend); when JNK above 50d SMA but flat/negative 10d return hold "
    "SPY 97% (mild risk-on); when JNK below 50d SMA hold TLT 97%; weekly rebalance; "
    "3-tier credit regime gating IWM+QQQ blend is distinct from existing 2-state JNK/VIX switchers"
)

STRATEGY = JnkTier3IwmQqqBlend()
