"""Credit spread regime equity allocation strategy.

Hypothesis: JNK/LQD ratio trend signals credit market risk appetite.
When high-yield bonds (JNK) are outperforming investment-grade (LQD) on a
trend basis, credit spreads are tightening (risk-on). Route this regime
signal to equity/macro ETFs rather than the credit ETFs themselves.

Signal: JNK/LQD ratio 20d MA vs 60d MA
  - Tightening spreads (20d MA >= 60d MA): QQQ 97% (risk-on growth)
  - Widening spreads (20d MA < 60d MA): TLT 60% + GLD 37% (macro defensives)
  - Rebalance weekly (every 5 bars)

Rationale: The JNK/LQD ratio is a well-established credit pulse indicator.
gen5_credit_spread_hyg_lqd (IS Calmar 0.67) already uses this signal but
allocates to JNK vs LQD themselves (staying in credit space). This strategy
routes the SAME signal to equity (QQQ) vs macro hedges (TLT + GLD), creating
a fundamentally different daily return path:
  - In risk-on: QQQ typically exceeds JNK by >5% annually
  - In risk-off: GLD adds non-correlated protection vs pure TLT

Diversification vs leaderboard:
  - gen5_credit_spread_hyg_lqd: routes to JNK/LQD (credit ETFs) — this
    routes to QQQ/TLT/GLD (equity + macro), so daily corr is much lower
  - No existing strategy combines credit-spread signal with QQQ allocation
  - GLD component as crisis hedge adds decorrelation in tail events
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 20     # 20d MA of JNK/LQD ratio
SLOW_MA = 60     # 60d MA of JNK/LQD ratio
REBALANCE = 5    # weekly
EXPOSURE = 0.97


class CreditSpreadQQQAlloc(Strategy):
    """JNK/LQD credit regime: tightening -> QQQ 97%; widening -> TLT 60% + GLD 37%."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Read JNK history
        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return []
        if jnk_hist is None or len(jnk_hist) < self.slow_ma + 2:
            return []
        jnk_close = jnk_hist["close"].dropna()

        # Read LQD history
        try:
            lqd_hist = ctx.history("LQD")
        except KeyError:
            return []
        if lqd_hist is None or len(lqd_hist) < self.slow_ma + 2:
            return []
        lqd_close = lqd_hist["close"].dropna()

        # Compute JNK/LQD ratio using aligned data
        # Use the shorter series length
        n = min(len(jnk_close), len(lqd_close))
        if n < self.slow_ma + 2:
            return []

        jnk_aligned = jnk_close.iloc[-n:].values
        lqd_aligned = lqd_close.iloc[-n:].values

        # Avoid division by zero
        if (lqd_aligned == 0).any():
            return []

        ratio = jnk_aligned / lqd_aligned

        # Compute MAs on the ratio
        if len(ratio) < self.slow_ma:
            return []

        fast_val = float(np.mean(ratio[-self.fast_ma:]))
        slow_val = float(np.mean(ratio[-self.slow_ma:]))

        if not np.isfinite(fast_val) or not np.isfinite(slow_val):
            return []

        # Regime: JNK outperforming LQD = tightening spreads = risk-on
        risk_on = fast_val >= slow_val

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target allocation
        if risk_on:
            # Tightening spreads: QQQ
            allocations = [("QQQ", self.exposure)]
        else:
            # Widening spreads: TLT + GLD defensive
            allocations = [("TLT", 0.60 * self.exposure), ("GLD", 0.40 * self.exposure)]

        target: dict[str, float] = {}
        for sym, weight in allocations:
            if sym in closes_now.index:
                target[sym] = weight

        # Fallback to SPY if target ETFs missing
        if not target:
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
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


NAME = "credit_spread_qqq_alloc"
HYPOTHESIS = (
    "Credit spread regime equity allocation: use JNK/LQD ratio 20d vs 60d MA crossover "
    "as risk regime; tightening spreads (JNK leading) hold QQQ 97%; widening spreads "
    "hold TLT 60%+GLD 37%; same JNK/LQD signal as gen5_credit_spread but routes to "
    "QQQ vs macro assets instead of credit ETFs; weekly rebalance"
)

UNIVERSE = ["JNK", "LQD", "QQQ", "TLT", "GLD"]

STRATEGY = CreditSpreadQQQAlloc()
