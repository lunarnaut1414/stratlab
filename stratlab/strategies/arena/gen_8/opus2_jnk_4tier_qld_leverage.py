"""JNK 4-Tier Credit Regime with QLD Leverage Sleeve — gen_8 opus-2 (gap_finder)

Hypothesis: Existing JNK strategies on the leaderboard top out at QQQ or SPY
in their bullish tier. None use a LEVERAGED equity sleeve as the
"super-bullish" tier. This strategy adds a 4th tier on top of the standard
3-tier JNK credit regime:

  - Tier A (super-bullish, JNK 60d MA AND JNK 21d return > +1%)
       QLD 50% + SPY 47%  (mild 2x QQQ tilt)
  - Tier B (bullish, JNK 60d MA but flat 21d ±1%)
       QQQ 97%
  - Tier C (weakening, JNK 60d MA AND JNK 21d return < -1%)
       SPY 60% + TLT 37%
  - Tier D (bearish, JNK < 60d MA)
       TLT 60% + IEF 37%

The key novelty is decomposing the "credit-healthy" regime into two
sub-regimes based on JNK 21d return: strong-and-rising vs strong-but-
weakening. Existing strategies use only the level (above/below MA) to
classify credit health.

Why this fills a gap:
- gen_6 gen8_jnk_xly_triple_gate_sp500 (0.75 Calmar) uses 3 tiers but
  no leverage anywhere.
- gen_8 jnk_tier3_iwm_qqq_blend (0.58 Calmar) uses IWM+QQQ at top tier
  but no leverage; h1=0.40 h2=1.27 is severely h2-dominated.
- gen_5 leveraged_etf_momentum used SSO (2x SPY) but with a TREND signal,
  not credit. SSO weight 0.6 too.
- The "JNK is strong AND accelerating" condition is rare enough (~15-25%
  of IS days) that the leverage tilt doesn't dominate; the QQQ tier still
  carries most of the return.

Weekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
JNK_MA = 60
JNK_LOOKBACK = 21
JNK_STRONG_RET = 0.01    # +1% 21d
JNK_WEAK_RET = -0.01     # -1% 21d
EXPOSURE = 0.97

_SPY = "SPY"
_QQQ = "QQQ"
_QLD = "QLD"
_TLT = "TLT"
_IEF = "IEF"
_JNK = "JNK"


class JNK4TierQLDLeverage(Strategy):
    """4-tier JNK credit regime with leveraged QLD top tier."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        jnk_lookback: int = JNK_LOOKBACK,
        jnk_strong_ret: float = JNK_STRONG_RET,
        jnk_weak_ret: float = JNK_WEAK_RET,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            jnk_lookback=jnk_lookback,
            jnk_strong_ret=jnk_strong_ret,
            jnk_weak_ret=jnk_weak_ret,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.jnk_lookback = int(jnk_lookback)
        self.jnk_strong_ret = float(jnk_strong_ret)
        self.jnk_weak_ret = float(jnk_weak_ret)
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
        jnk_ret = float(jnk_close.iloc[-1] / jnk_close.iloc[-self.jnk_lookback - 1] - 1.0)

        jnk_healthy = jnk_now > jnk_sma

        # Determine tier
        target: dict[str, float] = {}
        if not jnk_healthy:
            # Tier D — bearish credit
            if _TLT in live:
                target[_TLT] = self.exposure * 0.62
            if _IEF in live:
                target[_IEF] = self.exposure * 0.38
        elif jnk_ret > self.jnk_strong_ret:
            # Tier A — super-bullish (level + accelerating)
            if _QLD in live:
                target[_QLD] = self.exposure * 0.52
            if _SPY in live:
                target[_SPY] = self.exposure * 0.48
        elif jnk_ret < self.jnk_weak_ret:
            # Tier C — weakening despite level
            if _SPY in live:
                target[_SPY] = self.exposure * 0.62
            if _TLT in live:
                target[_TLT] = self.exposure * 0.38
        else:
            # Tier B — bullish steady
            if _QQQ in live:
                target[_QQQ] = self.exposure

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
    return [_SPY, _QQQ, _QLD, _TLT, _IEF, _JNK]


NAME = "opus2_jnk_4tier_qld_leverage"
HYPOTHESIS = (
    "JNK 4-tier credit regime with QLD (2x QQQ) leverage sleeve in super-bullish tier: "
    "JNK>60d SMA AND JNK 21d return >+1% hold QLD 50%+SPY 47% (mild leverage); "
    "JNK>60d SMA AND 21d return -1 to +1% hold QQQ 97%; JNK>60d SMA but 21d <-1% "
    "hold SPY 60%+TLT 37%; JNK<60d SMA hold TLT 60%+IEF 37%. Weekly rebalance. "
    "Novel decomposition: existing JNK tiers use level only; this adds a "
    "level+acceleration combination to unlock leveraged sleeve in strongest tier."
)

UNIVERSE = _universe

STRATEGY = JNK4TierQLDLeverage()
