"""opus-2 gap_finder: REIT subsegment regime (VNQ vs REM).

Gap identified: the brief explicitly lists REIT subsegmentation (VNQ vs IYR vs
REM) as an untouched frontier. No prior strategy uses mortgage REITs (REM) at
all. REM is structurally different from equity REITs: it's leveraged exposure
to mortgage spreads + duration. When REM outperforms VNQ, it signals
*spread tightening* in agency MBS — risk-on macro condition. When VNQ
outperforms REM, it signals duration stress or credit-spread widening — defensive.

This is orthogonal to JNK/LQD credit z-score (5 variants on leaderboard)
because REM trades agency MBS spread, not corporate credit. Both VNQ (since
2004) and REM (since 2007) cover the IS window.

Hypothesis: 90d (REM - VNQ) return spread:
  - positive  -> agency MBS spreads tightening -> macro risk-on -> 92% QQQ
  - negative  -> spreads widening or duration stress -> defensive 50% SPY + 45% SHY
  - very negative (REM_lag > 10pp) -> deeper stress -> 60% SHY + 30% TLT

Universe: VNQ, REM, QQQ, SPY, SHY, TLT
Rebalance every 10 bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["VNQ", "REM", "QQQ", "SPY", "SHY", "TLT"]

LOOKBACK = 90
REBALANCE_EVERY = 10
STRESS_THRESHOLD = -0.10  # REM lags VNQ by >10pp = deeper stress


def _return_n(closes: pd.Series, n: int) -> float:
    c = closes.dropna()
    if len(c) < n + 1:
        return float("nan")
    return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1.0)


class ReitSubsegmentRegime(Strategy):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        rebalance_every: int = REBALANCE_EVERY,
        stress_threshold: float = STRESS_THRESHOLD,
    ) -> None:
        super().__init__(
            lookback=lookback,
            rebalance_every=rebalance_every,
            stress_threshold=stress_threshold,
        )
        self.lookback = int(lookback)
        self.rebalance_every = int(rebalance_every)
        self.stress_threshold = float(stress_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.lookback + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        try:
            vnq = ctx.history("VNQ")
            rem = ctx.history("REM")
        except KeyError:
            return []
        if vnq is None or rem is None:
            return []
        if len(vnq) < self.lookback + 5 or len(rem) < self.lookback + 5:
            return []

        vnq_r = _return_n(vnq["close"], self.lookback)
        rem_r = _return_n(rem["close"], self.lookback)
        if not np.isfinite(vnq_r) or not np.isfinite(rem_r):
            return []

        spread = rem_r - vnq_r

        if spread > 0:
            # MBS spreads tightening -> macro risk-on
            target = {"QQQ": 0.92}
        elif spread > self.stress_threshold:
            # Mild stress / VNQ leadership -> SPY tilt + cash
            target = {"SPY": 0.55, "SHY": 0.40}
        else:
            # Deep stress (REM trailing by >10pp) -> defensive
            target = {"SHY": 0.60, "TLT": 0.30}

        live = ctx.closes()
        if live.empty:
            return []
        live_dict = {s: float(p) for s, p in live.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))

        for sym, weight in target.items():
            price = live_dict.get(sym)
            if price is None or price <= 0:
                continue
            target_shares = int(equity * weight / price)
            cur_shares = int(ctx.position(sym).size)
            delta = target_shares - cur_shares
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


NAME = "opus2_reit_subsegment_regime"
HYPOTHESIS = (
    "REIT subsegment regime: 90d (REM - VNQ) return spread as agency MBS spread proxy. "
    "spread>0 -> 92% QQQ (MBS spreads tightening, risk-on); spread<=0 -> SPY+SHY defensive; "
    "spread<-10pp -> SHY+TLT (deep stress). Orthogonal to corporate-credit JNK/LQD signals; "
    "REM (mortgage REITs) untouched after 4 rounds."
)

STRATEGY = ReitSubsegmentRegime()
