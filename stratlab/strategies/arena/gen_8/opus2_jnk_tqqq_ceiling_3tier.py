"""JNK 3-Tier Credit with TQQQ Leverage Ceiling — gen_8 opus-2 (gap_finder)

Hypothesis: A tight, infrequent leverage trigger using TQQQ (3x QQQ) at a small
allocation (30%) plus a large SPY core (67%) gives leveraged equity exposure
ONLY when credit conditions are unambiguously strong. The 3x decay/path-
dependence of TQQQ is mitigated by:
  - Small allocation (30% not 50-100%)
  - Tight trigger (JNK > 50d MA AND 21d return > +2% — rare)
  - SPY core (67%) provides stable base when TQQQ chops

3-tier rotation:
  - Tier A (JNK > 50d MA AND JNK 21d > +2%): TQQQ 30% + SPY 67%
       Strong, accelerating credit — modest 3x QQQ overlay
  - Tier B (JNK > 50d MA otherwise):         SPY 97%
       Healthy credit — pure SPY exposure
  - Tier C (JNK < 50d MA):                   TLT 60% + IEF 37%
       Credit stress — defensive bonds

Why distinct from existing JNK strategies:
- gen_6 hy_credit_qqq_rotation routes JNK-strong → QQQ 97% (no leverage)
- gen_8 jnk_xly_triple_gate uses 3-signal vote, not credit-acceleration
- gen_8 jnk_tier3_iwm_qqq_blend uses 3-tier IWM+QQQ blend at top
- gen_5 leveraged_etf_momentum used SSO with TREND signals, not credit
- opus-2 jnk_4tier_qld_leverage (just-accepted) uses QLD (2x) not TQQQ (3x)
  AND uses QQQ base not SPY base AND uses 4 tiers not 3
- The TQQQ-as-overlay-on-SPY pattern is structurally unique on the board

Weekly rebalance. TQQQ cached from 2010-02 — covers most of IS window;
strategy gracefully degrades to SPY when TQQQ missing.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
JNK_MA = 50
JNK_LOOKBACK = 21
JNK_HOT_RET = 0.02
W_HOT_TQQQ = 0.30
W_HOT_SPY = 0.67
W_HEALTHY_SPY = 0.97
W_STRESS_TLT = 0.60
W_STRESS_IEF = 0.37
EXPOSURE = 0.97

_SPY = "SPY"
_TQQQ = "TQQQ"
_TLT = "TLT"
_IEF = "IEF"
_JNK = "JNK"


class JNKTqqqCeiling3Tier(Strategy):
    """3-tier JNK credit with TQQQ leverage ceiling on SPY core."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        jnk_lookback: int = JNK_LOOKBACK,
        jnk_hot_ret: float = JNK_HOT_RET,
        w_hot_tqqq: float = W_HOT_TQQQ,
        w_hot_spy: float = W_HOT_SPY,
        w_healthy_spy: float = W_HEALTHY_SPY,
        w_stress_tlt: float = W_STRESS_TLT,
        w_stress_ief: float = W_STRESS_IEF,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            jnk_lookback=jnk_lookback,
            jnk_hot_ret=jnk_hot_ret,
            w_hot_tqqq=w_hot_tqqq,
            w_hot_spy=w_hot_spy,
            w_healthy_spy=w_healthy_spy,
            w_stress_tlt=w_stress_tlt,
            w_stress_ief=w_stress_ief,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.jnk_lookback = int(jnk_lookback)
        self.jnk_hot_ret = float(jnk_hot_ret)
        self.w_hot_tqqq = float(w_hot_tqqq)
        self.w_hot_spy = float(w_hot_spy)
        self.w_healthy_spy = float(w_healthy_spy)
        self.w_stress_tlt = float(w_stress_tlt)
        self.w_stress_ief = float(w_stress_ief)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.jnk_lookback) + 10
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

        # JNK credit signals
        try:
            jnk_hist = ctx.history(_JNK)
        except KeyError:
            return []
        if jnk_hist is None:
            return []
        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) < max(self.jnk_ma, self.jnk_lookback) + 1:
            return []

        jnk_now = float(jnk_close.iloc[-1])
        jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
        jnk_ret = float(
            jnk_close.iloc[-1] / jnk_close.iloc[-self.jnk_lookback - 1] - 1.0
        )
        jnk_healthy = jnk_now > jnk_sma
        jnk_hot = jnk_healthy and jnk_ret > self.jnk_hot_ret

        target: dict[str, float] = {}
        if jnk_hot and _TQQQ in live:
            # Tier A — leveraged ceiling
            target[_TQQQ] = self.w_hot_tqqq
            if _SPY in live:
                target[_SPY] = self.w_hot_spy
        elif jnk_healthy:
            # Tier B — healthy SPY
            if _SPY in live:
                target[_SPY] = self.w_healthy_spy
        else:
            # Tier C — stress defensive
            if _TLT in live:
                target[_TLT] = self.w_stress_tlt
            if _IEF in live:
                target[_IEF] = self.w_stress_ief

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
    return [_SPY, _TQQQ, _TLT, _IEF, _JNK]


NAME = "opus2_jnk_tqqq_ceiling_3tier"
HYPOTHESIS = (
    "JNK 3-tier credit with TQQQ (3x QQQ) leverage ceiling on SPY core: JNK above 50d "
    "SMA AND JNK 21d return > +2% (rare hot signal) hold TQQQ 30%+SPY 67% (modest 3x "
    "overlay); JNK above 50d SMA otherwise hold SPY 97%; JNK below 50d SMA hold "
    "TLT 60%+IEF 37%. Weekly rebalance. Distinct from JNK 4-tier QLD (2x vs 3x; SPY "
    "vs QQQ base; tighter trigger), and from existing JNK-MA gates that have no leverage."
)

UNIVERSE = _universe

STRATEGY = JNKTqqqCeiling3Tier()
