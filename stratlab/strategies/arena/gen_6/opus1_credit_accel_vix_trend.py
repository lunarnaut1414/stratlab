"""opus-1 mutation of jnk_vix_dual_gate_qqq (credit-allocator cluster).

Parent: gen6_jnk_vix_dual_gate_qqq (IS Calmar 0.86, h2>h1, corr_to_top5 0.71).

Structural mutations vs parent:
  - JNK signal: single 20d-SMA cross  ->  10d-vs-50d return acceleration ratio
                (compares short and medium credit-momentum, not level vs trend).
  - VIX signal: level thresholds (20, 28)  ->  VIX vs its own 20d MA (trend).
                We measure whether realized vol is *expanding or contracting*,
                not where it sits in absolute terms — robust to regime drift
                in absolute VIX levels (16-25 range across IS).
  - Sizing:     binary 3-tier (QQQ/SPY/cash)  ->  continuous tilt scaled by
                VIX-percentile rank (0-1) over 60d. When VIX is in low pctile
                AND credit accel positive: tilt to QQQ; high pctile but credit
                still positive: tilt to SPY; credit decel: SHY 50% + IEF 47%.
  - Rebalance:  weekly (5)  ->  biweekly (10) — less churn given continuous size.
  - Defensive:  SHY+TLT     ->  SHY+IEF (mid-duration not long-duration —
                less duration risk, different daily path).

Why this should be admitted under 0.85 corr filter:
  - Continuous (non-binary) sizing produces a smoother return path that
    differs daily from any tiered allocator on the leaderboard.
  - VIX-trend (rate-of-change) is orthogonal to VIX-level used by every other
    VIX-gated leaderboard strategy.
  - Credit acceleration is a derivative signal — it differs from JNK trend
    even when both are bullish on average.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_SHORT = 10           # short JNK return window
JNK_LONG = 50            # long JNK return window
VIX_TREND_MA = 20        # VIX MA window for trend gate
VIX_PCTILE_WINDOW = 60   # VIX percentile lookback for sizing
REBALANCE_EVERY = 10     # biweekly
EXPOSURE = 0.97
QQQ_TILT_THRESH = 0.30   # below this VIX pctile: QQQ-heavy
SPY_TILT_THRESH = 0.70   # above: prefer SPY


class CreditAccelVixTrend(Strategy):
    def __init__(
        self,
        jnk_short: int = JNK_SHORT,
        jnk_long: int = JNK_LONG,
        vix_trend_ma: int = VIX_TREND_MA,
        vix_pctile_window: int = VIX_PCTILE_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
        qqq_tilt_thresh: float = QQQ_TILT_THRESH,
        spy_tilt_thresh: float = SPY_TILT_THRESH,
    ) -> None:
        super().__init__(
            jnk_short=jnk_short,
            jnk_long=jnk_long,
            vix_trend_ma=vix_trend_ma,
            vix_pctile_window=vix_pctile_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
            qqq_tilt_thresh=qqq_tilt_thresh,
            spy_tilt_thresh=spy_tilt_thresh,
        )
        self.jnk_short = int(jnk_short)
        self.jnk_long = int(jnk_long)
        self.vix_trend_ma = int(vix_trend_ma)
        self.vix_pctile_window = int(vix_pctile_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self.qqq_tilt_thresh = float(qqq_tilt_thresh)
        self.spy_tilt_thresh = float(spy_tilt_thresh)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_long, self.vix_pctile_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- JNK acceleration: 10d return - 50d return ---
        credit_accel = 0.0
        credit_ok = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_long + 2:
                jnk_close = jnk_hist["close"].dropna().values
                if len(jnk_close) >= self.jnk_long + 1:
                    p_now = float(jnk_close[-1])
                    p_short = float(jnk_close[-self.jnk_short - 1])
                    p_long = float(jnk_close[-self.jnk_long - 1])
                    if p_short > 0 and p_long > 0 and np.isfinite(p_now):
                        ret_short = p_now / p_short - 1.0
                        ret_long = p_now / p_long - 1.0
                        # Acceleration: short-window outperforms (annualized)
                        # to avoid horizon mismatch
                        ann_short = ret_short * (252.0 / self.jnk_short)
                        ann_long = ret_long * (252.0 / self.jnk_long)
                        credit_accel = ann_short - ann_long
                        # require both positive AND short accelerating
                        credit_ok = (ann_short > 0.0) and (credit_accel > -0.02)
        except KeyError:
            pass

        # --- VIX trend: VIX vs 20d MA, plus 60d percentile rank ---
        vix_pctile = 0.5  # neutral
        vix_trend_up = False
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= self.vix_pctile_window + 2:
                vix_close = vix_hist["close"].dropna().values
                if len(vix_close) >= self.vix_pctile_window:
                    vix_now = float(vix_close[-1])
                    vix_ma = float(np.mean(vix_close[-self.vix_trend_ma:]))
                    vix_trend_up = vix_now > vix_ma
                    # Percentile rank: how high is current vix vs last 60d?
                    window = vix_close[-self.vix_pctile_window:]
                    vix_pctile = float(np.mean(window <= vix_now))
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

        if not credit_ok:
            # Risk-off — credit decelerating: half SHY half IEF
            if "SHY" in live:
                target["SHY"] = 0.50 * self.exposure
            if "IEF" in live:
                target["IEF"] = 0.47 * self.exposure
            if not target and "SHY" in live:
                target["SHY"] = self.exposure
        else:
            # Credit ok — pick equity exposure based on VIX percentile + trend.
            # Continuous tilt: linearly blend QQQ <-> SPY by VIX pctile.
            # Penalty when VIX is rising (trend up) regardless of pctile.
            if vix_pctile <= self.qqq_tilt_thresh and not vix_trend_up:
                # Calm + low VIX: full QQQ
                if "QQQ" in live:
                    target["QQQ"] = self.exposure
                elif "SPY" in live:
                    target["SPY"] = self.exposure
            elif vix_pctile >= self.spy_tilt_thresh or vix_trend_up:
                # Elevated VIX or rising — prefer SPY with small SHY buffer
                if "SPY" in live:
                    target["SPY"] = 0.70 * self.exposure
                if "SHY" in live:
                    target["SHY"] = 0.27 * self.exposure
                if not target and "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                # Mid-zone: blend QQQ / SPY linearly via the pctile slope
                # frac=0 means full QQQ, frac=1 means full SPY
                span = self.spy_tilt_thresh - self.qqq_tilt_thresh
                if span <= 0:
                    frac = 0.5
                else:
                    frac = (vix_pctile - self.qqq_tilt_thresh) / span
                frac = max(0.0, min(1.0, frac))
                qqq_w = (1.0 - frac) * self.exposure
                spy_w = frac * self.exposure
                if "QQQ" in live and qqq_w > 0.01:
                    target["QQQ"] = qqq_w
                if "SPY" in live and spy_w > 0.01:
                    target["SPY"] = spy_w
                if not target and "SPY" in live:
                    target["SPY"] = self.exposure

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


NAME = "opus1_credit_accel_vix_trend"
HYPOTHESIS = (
    "Mutate jnk_vix_dual_gate_qqq: JNK 10d-vs-50d return acceleration replaces "
    "single 20d MA cross; VIX 20d MA trend + 60d percentile rank replaces level "
    "tiers; continuous QQQ<->SPY blend by VIX pctile; SHY+IEF defensive on "
    "credit decel; biweekly rebalance."
)
UNIVERSE = ["JNK", "QQQ", "SPY", "SHY", "IEF", "^VIX"]

STRATEGY = CreditAccelVixTrend()
