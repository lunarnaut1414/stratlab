"""opus-1 mutation of jnk_vix_dual_gate_qqq (credit-allocator cluster).

Parent: gen6_jnk_vix_dual_gate_qqq (IS Calmar 0.86, h2>h1, corr_to_top5 0.71).

Structural mutations vs parent:
  - VIX gate:   level thresholds (calm <20, caution <28)  ->  VIX 5d
                rate-of-change gate (negative ROC = vol contracting,
                positive ROC = vol expanding).
                Tier 1 (QQQ): JNK > 50d MA AND VIX-5d-change < -1.0
                Tier 2 (SPY): JNK > 50d MA AND VIX-5d-change in [-1, +1]
                Risk-off:     JNK < 50d MA OR VIX-5d-change > +2.0
  - JNK MA:     20  ->  50 (slower MA = fewer false credit signals; the
                shorter 20d MA lines up with several leaderboard variants).
  - Rebalance:  weekly (5)  ->  biweekly (10) — slower MA wants slower
                rebalance.
  - Defensive:  SHY 50% + TLT 47%  ->  SHY 47% + AGG 50% (mid-duration
                rather than long-duration; different daily innovations).

Why this should be admitted under 0.85 corr filter:
  - VIX rate-of-change is a momentum-derivative, structurally different
    from VIX level. The 5d ROC fires on entirely different days than
    'VIX < 20' (e.g., VIX could be 17 but rising from 14 = elevated
    danger to ROC, calm to level).
  - JNK 50d MA is materially slower than parent's 20d MA — the credit
    state-flip dates are different.
  - AGG (mid-duration) replaces TLT (long-duration) — different daily
    PnL through 2013 taper tantrum and 2018 rate selloff.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 50              # JNK trend (slower than parent's 20)
VIX_ROC_WINDOW = 5       # VIX rate-of-change lookback
VIX_ROC_CALM = -1.0      # below this VIX-pt ROC: vol contracting = QQQ
VIX_ROC_PANIC = 2.0      # above this VIX-pt ROC: panic = risk-off override
REBALANCE_EVERY = 10     # biweekly
EXPOSURE = 0.97


class JnkVixRocGate(Strategy):
    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        vix_roc_window: int = VIX_ROC_WINDOW,
        vix_roc_calm: float = VIX_ROC_CALM,
        vix_roc_panic: float = VIX_ROC_PANIC,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            vix_roc_window=vix_roc_window,
            vix_roc_calm=vix_roc_calm,
            vix_roc_panic=vix_roc_panic,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.vix_roc_window = int(vix_roc_window)
        self.vix_roc_calm = float(vix_roc_calm)
        self.vix_roc_panic = float(vix_roc_panic)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.jnk_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK trend (50d SMA)
        credit_bullish = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna().values
                if len(jnk_close) >= self.jnk_ma + 1:
                    jnk_now = float(jnk_close[-1])
                    jnk_ma_val = float(np.mean(jnk_close[-self.jnk_ma:]))
                    credit_bullish = jnk_now > jnk_ma_val
        except KeyError:
            pass

        # VIX 5d rate-of-change (today - 5d ago, in VIX points)
        vix_roc = 0.0
        have_vix = False
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= self.vix_roc_window + 2:
                vix_close = vix_hist["close"].dropna().values
                if len(vix_close) >= self.vix_roc_window + 1:
                    vix_now = float(vix_close[-1])
                    vix_then = float(vix_close[-self.vix_roc_window - 1])
                    vix_roc = vix_now - vix_then
                    have_vix = True
        except KeyError:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Risk-off triggers: credit weak OR VIX panic spike
        risk_off = (not credit_bullish) or (have_vix and vix_roc > self.vix_roc_panic)

        if risk_off:
            if "SHY" in live:
                target["SHY"] = 0.47 * self.exposure
            if "AGG" in live:
                target["AGG"] = 0.50 * self.exposure
            elif "IEF" in live:
                target["IEF"] = 0.50 * self.exposure
            if not target and "SHY" in live:
                target["SHY"] = self.exposure
        elif have_vix and vix_roc < self.vix_roc_calm:
            # Tier 1: credit good + vol contracting → QQQ
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Tier 2: credit good + neutral vol → SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure

        if not target:
            return []

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


NAME = "opus1_jnk_vix_roc_gate"
HYPOTHESIS = (
    "Mutate jnk_vix_dual_gate_qqq: VIX 5d rate-of-change replaces VIX level "
    "thresholds (-1=calm/QQQ, +2=panic/risk-off); JNK 50d MA replaces 20d "
    "MA; SHY+AGG defensive replaces SHY+TLT; biweekly rebalance."
)
UNIVERSE = ["JNK", "QQQ", "SPY", "SHY", "AGG", "IEF", "TLT", "^VIX"]

STRATEGY = JnkVixRocGate()
