"""Absolute Momentum QQQ/GLD/TLT 3-Asset Rotation — gen_8 sonnet-4

Hypothesis: Rotate among 3 assets using absolute momentum + credit confirmation:
  1. Hold QQQ when QQQ 63d return > 0 AND JNK above 30d MA
     (equity momentum + credit confirmation both positive)
  2. Hold GLD when QQQ condition not met but GLD 63d return > 0
     (equity not trending up but gold is in uptrend — inflation/risk-off gold rally)
  3. Hold TLT otherwise (defensive bond allocation)

Weekly rebalance. The 3-layer waterfall selects the best available risk-on asset
given prevailing credit and momentum conditions.

Rationale: In 2010-2018 IS window:
  - QQQ had strong 63d positive returns most of the time (bull market) → high exposure
  - When QQQ faltered (2011, 2015-16, 2018), JNK often broke its 30d MA first → avoid equity dips
  - GLD catches the few periods where inflation fears drove gold up vs equity down (2011-12)
  - TLT as ultimate safety net

Distinction from existing strategies:
- gen6_hy_credit_qqq_rotation: JNK+SPY 100d SMA → QQQ/TLT (no GLD intermediate rung)
- gen6_jnk_vix_dual_gate_qqq: JNK+VIX dual gate (not absolute momentum on QQQ itself)
- gen6_jnk_continuous_credit_tilt: continuous weighting (not discrete rotation)
- This uses QQQ's OWN 63d absolute momentum as primary signal, with JNK as credit gate,
  and GLD as an intermediate safe-haven tier between equity and bonds.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
MOMENTUM_WINDOW = 63       # QQQ and GLD absolute momentum
JNK_MA = 30                # JNK credit gate
EXPOSURE = 0.97


class QqqGldTltAbsMomentum(Strategy):
    """3-asset waterfall: QQQ (trend+credit) → GLD (inflation safe-haven) → TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        jnk_ma: int = JNK_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            jnk_ma=jnk_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.jnk_ma = int(jnk_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.jnk_ma) + 5
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

        # QQQ 63d absolute momentum
        qqq_trend = False
        try:
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) >= self.momentum_window + 1 and "QQQ" in prices.columns:
                col = prices["QQQ"].dropna()
                if len(col) >= self.momentum_window + 1:
                    p_start = float(col.iloc[-self.momentum_window])
                    p_end = float(col.iloc[-1])
                    if p_start > 0:
                        qqq_ret = p_end / p_start - 1.0
                        qqq_trend = np.isfinite(qqq_ret) and qqq_ret > 0
        except Exception:
            pass

        # JNK credit confirmation
        jnk_credit_ok = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_credit_ok = jnk_now > jnk_ma_val
        except Exception:
            pass

        # GLD 63d absolute momentum
        gld_trend = False
        try:
            if len(prices) >= self.momentum_window + 1 and "GLD" in prices.columns:
                col_gld = prices["GLD"].dropna()
                if len(col_gld) >= self.momentum_window + 1:
                    p_start_gld = float(col_gld.iloc[-self.momentum_window])
                    p_end_gld = float(col_gld.iloc[-1])
                    if p_start_gld > 0:
                        gld_ret = p_end_gld / p_start_gld - 1.0
                        gld_trend = np.isfinite(gld_ret) and gld_ret > 0
        except Exception:
            pass

        # Waterfall selection
        target: dict[str, float] = {}
        if qqq_trend and jnk_credit_ok and "QQQ" in live:
            target["QQQ"] = self.exposure
        elif gld_trend and "GLD" in live:
            target["GLD"] = self.exposure
        elif "TLT" in live:
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


UNIVERSE = ["QQQ", "SPY", "TLT", "GLD", "JNK"]

NAME = "qqq_gld_tlt_abs_momentum"
HYPOTHESIS = (
    "Absolute momentum QQQ/GLD/TLT 3-asset rotation: hold QQQ when QQQ 63d return > 0 "
    "AND JNK above 30d MA (trend + credit confirmation); hold GLD when QQQ condition not "
    "met but GLD 63d return positive; hold TLT otherwise; weekly rebalance; 3-asset "
    "absolute momentum distinct from SPY-centric momentum strategies"
)

STRATEGY = QqqGldTltAbsMomentum()
