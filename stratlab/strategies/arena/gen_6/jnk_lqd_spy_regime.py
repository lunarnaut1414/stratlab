"""JNK/LQD MA crossover credit regime + tiered SPY/SSO exposure.

Hypothesis: Credit spread regime as primary signal:
  - Risk-on (JNK 20d MA > 60d MA, tightening spreads): boosted equity
    exposure using SSO (2x SPY) 40% + SPY 55% = ~1.35x net SPY exposure.
  - Neutral (within threshold): base SPY 95%.
  - Risk-off (JNK 20d MA < 60d MA, widening spreads): reduced equity
    SPY 30% + TLT 60%.

Rebalance every 5 bars (weekly).

Key distinctions from existing leaderboard:
  - gen5_credit_spread_hyg_lqd: uses JNK/LQD (same signal) but only
    rotates between JNK and LQD bonds — does not hold equities.
  - gen5_bond_equity_regime: TLT/SPY ratio MA crossover (not JNK/LQD).
  - gen5_leveraged_etf_momentum: SSO + 200d SMA + momentum gates (3 gates).
    This strategy uses only credit spread as the single gate.
  - The credit-to-equity mapping (risk-on -> leverage, risk-off -> bonds)
    is a fundamentally different pathway.

Rationale:
  - JNK/LQD 20d/60d MA crossover is slower than 20d raw return, filtering
    noise. In the 2010-2018 IS window, credit was mostly in risk-on mode
    which captures the equity bull run.
  - SSO (2x SPY) in the top risk-on bucket captures leveraged upside
    during confirmed credit-friendly regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
FAST_MA = 20             # JNK fast MA
SLOW_MA = 60             # JNK slow MA
EXPOSURE = 0.97

_JNK = "JNK"
_SPY = "SPY"
_SSO = "SSO"   # 2x SPY leveraged ETF
_TLT = "TLT"

# Risk-on: SSO 40% + SPY 55%
# Neutral: SPY 95%
# Risk-off: SPY 30% + TLT 60%
RISK_ON = {_SSO: 0.40, _SPY: 0.55}
NEUTRAL = {_SPY: 0.95}
RISK_OFF = {_SPY: 0.30, _TLT: 0.60}


class JnkLqdSpyRegime(Strategy):
    """JNK/LQD credit spread regime + tiered SPY/SSO/TLT allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Compute JNK MA crossover for credit regime
        regime = "neutral"
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.slow_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.slow_ma:
                    fast = float(jnk_close.iloc[-self.fast_ma:].mean())
                    slow = float(jnk_close.iloc[-self.slow_ma:].mean())
                    if np.isfinite(fast) and np.isfinite(slow) and slow > 0:
                        if fast > slow:
                            regime = "risk_on"
                        else:
                            regime = "risk_off"
        except (KeyError, IndexError):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Select target allocation based on regime
        if regime == "risk_on":
            raw_target = RISK_ON
        elif regime == "risk_off":
            raw_target = RISK_OFF
        else:
            raw_target = NEUTRAL

        # Scale by exposure and filter out missing symbols
        target: dict[str, float] = {}
        for sym, w in raw_target.items():
            if sym in closes_now.index:
                target[sym] = w * self.exposure

        # If SSO not available (fallback), replace SSO with SPY
        if _SSO not in closes_now.index and _SSO in target:
            spy_w = target.pop(_SSO)
            target[_SPY] = target.get(_SPY, 0) + spy_w

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
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


UNIVERSE = [_JNK, _SPY, _SSO, _TLT]

NAME = "jnk_lqd_spy_regime"
HYPOTHESIS = (
    "SPY long-only tiered exposure via JNK 20d/60d MA crossover credit regime: "
    "risk-on (JNK fast>slow) = SSO 40%+SPY 55%; neutral = SPY 95%; "
    "risk-off (JNK fast<slow) = SPY 30%+TLT 60%; weekly rebalance"
)

STRATEGY = JnkLqdSpyRegime()
