"""Multi-Credit-Signal Composite Gate QQQ/SPY/TLT — gen_8 sonnet-4

Hypothesis: Compute a 3-signal credit score using:
  1. JNK vs 30d MA (JNK above MA = risk-on)
  2. HYG vs 20d MA (HYG above MA = risk-on)
  3. LQD/SHY 20d return spread > 0 (investment-grade outperforming T-bills = risk-on)

Score = number of risk-on signals (0, 1, 2, or 3):
  - Score 3: all signals risk-on -> hold QQQ 97%
  - Score 2: two risk-on -> hold SPY 97%
  - Score 0 or 1: credit weakness -> hold TLT 97%

Biweekly rebalance. Multi-signal composite is more robust than single JNK/HYG
signal — requires consensus across HY, IG, and credit-vs-safety spreads.

Rationale: Each of the 3 credit signals measures a different dimension:
  - JNK: high-yield CCC/BB credit risk appetite
  - HYG: broader BB/B/CCC high-yield market momentum
  - LQD/SHY: investment-grade credit vs cash (risk-free) spread compression

Requiring all 3 to confirm risk-on before going to aggressive QQQ reduces
false positives that plague single-indicator credit strategies.

Distinction from existing strategies:
- gen6_jnk_vix_dual_gate_qqq: JNK + VIX (not 3 credit signals)
- gen6_hy_credit_qqq_rotation: single JNK + SPY trend
- gen6_jnk_continuous_credit_tilt: continuous tilt, not discrete 3-signal voting
- gen7_opus2_pff_jnk: PFF/JNK spread (credit quality), not multi-signal
- This uses 3 independent credit measures (HY trend, IG trend, IG vs cash)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # bi-weekly
JNK_MA = 30                # JNK slow MA window
HYG_MA = 20                # HYG fast MA window
LQD_SHY_WINDOW = 20        # LQD/SHY return comparison window
EXPOSURE = 0.97


class MultiCreditComposite(Strategy):
    """3-signal credit composite gating QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        hyg_ma: int = HYG_MA,
        lqd_shy_window: int = LQD_SHY_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            hyg_ma=hyg_ma,
            lqd_shy_window=lqd_shy_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.hyg_ma = int(hyg_ma)
        self.lqd_shy_window = int(lqd_shy_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.hyg_ma, self.lqd_shy_window) + 10
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

        # Signal 1: JNK vs 30d MA
        signal_jnk = 0
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    if jnk_now > jnk_ma_val:
                        signal_jnk = 1
        except Exception:
            pass

        # Signal 2: HYG vs 20d MA
        signal_hyg = 0
        try:
            hyg_hist = ctx.history("HYG")
            if hyg_hist is not None and len(hyg_hist) >= self.hyg_ma + 2:
                hyg_close = hyg_hist["close"].dropna()
                if len(hyg_close) >= self.hyg_ma:
                    hyg_ma_val = float(hyg_close.iloc[-self.hyg_ma:].mean())
                    hyg_now = float(hyg_close.iloc[-1])
                    if hyg_now > hyg_ma_val:
                        signal_hyg = 1
        except Exception:
            pass

        # Signal 3: LQD 20d return vs SHY 20d return (IG credit vs cash)
        signal_lqd = 0
        try:
            lqd_hist = ctx.history("LQD")
            shy_hist = ctx.history("SHY")
            if (lqd_hist is not None and len(lqd_hist) >= self.lqd_shy_window + 2 and
                    shy_hist is not None and len(shy_hist) >= self.lqd_shy_window + 2):
                lqd_close = lqd_hist["close"].dropna()
                shy_close = shy_hist["close"].dropna()
                if len(lqd_close) >= self.lqd_shy_window and len(shy_close) >= self.lqd_shy_window:
                    lqd_ret = float(lqd_close.iloc[-1] / lqd_close.iloc[-self.lqd_shy_window] - 1.0)
                    shy_ret = float(shy_close.iloc[-1] / shy_close.iloc[-self.lqd_shy_window] - 1.0)
                    if np.isfinite(lqd_ret) and np.isfinite(shy_ret) and lqd_ret > shy_ret:
                        signal_lqd = 1
        except Exception:
            pass

        # Composite credit score
        credit_score = signal_jnk + signal_hyg + signal_lqd

        # Route to appropriate ETF based on score
        target: dict[str, float] = {}
        if credit_score == 3:
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        elif credit_score == 2:
            if "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Score 0 or 1: credit weakness
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


UNIVERSE = ["QQQ", "SPY", "TLT", "JNK", "HYG", "LQD", "SHY"]

NAME = "multi_credit_composite"
HYPOTHESIS = (
    "Multi-credit-spread composite gating QQQ/SPY/TLT: compute 3-signal credit score "
    "(JNK vs 30d MA, HYG vs 20d MA, LQD/SHY 20d return spread); when all 3 signal risk-on "
    "hold QQQ 97%; when 2 of 3 risk-on hold SPY 97%; when 0-1 risk-on hold TLT 97%; "
    "biweekly rebalance; multi-signal credit composite distinct from single JNK/HYG-based strategies"
)

STRATEGY = MultiCreditComposite()
