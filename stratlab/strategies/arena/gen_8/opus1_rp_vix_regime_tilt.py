"""opus-1 / gen_8 — Risk Parity with VIX-Regime Tilt

Mutation of gen8_rp_yield_curve_tilt (IS Calmar 0.76, but SEVERE h1/h2 1.49/0.29).

Parent: inverse-vol RP on SPY/IEF/GLD with a 10Y-2Y yield curve slope tilt.
The yield-curve signal monotonically declined through IS (steep 2010-14,
flattening 2015-18), giving most of the strategy's alpha in h1 — severe
h2 weakness.

This variant keeps the same SPY/IEF/GLD inverse-vol base but swaps the macro
tilt to a VIX-regime tilt (a more stationary signal):
  - VIX 10d MA in bottom quartile of trailing 252d (low-vol regime,
    structurally calm): shift 15% of IEF allocation into SPY (growth tilt)
  - VIX 10d MA in top quartile of trailing 252d (high-vol regime, stress):
    shift 15% of SPY allocation into GLD (defensive tilt)
  - Otherwise: plain inverse-vol RP

Goal: more stable h1/h2 by using a self-normalizing signal that should
trigger roughly equally across both halves of IS (VIX cycles every 2-3 years).
Monthly rebalance, ETF-only universe.

Note on "VIX percentile" rather than absolute level: gen8_vix_gated_qqq_gld_blend
used absolute thresholds (16/22/30) and showed regime fragility (h2 0.39,
loss_mode_corr 0.91). Percentile rank is regime-adaptive.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21
VOL_WINDOW = 20
VIX_SMOOTH = 10
VIX_PCT_WINDOW = 252
VIX_LOW_PCT = 0.25
VIX_HIGH_PCT = 0.75
TILT_FRACTION = 0.15
EXPOSURE = 0.97


class RpVixRegimeTilt(Strategy):
    """Risk parity SPY/IEF/GLD with VIX-percentile-rank regime tilt."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        vix_smooth: int = VIX_SMOOTH,
        vix_pct_window: int = VIX_PCT_WINDOW,
        vix_low_pct: float = VIX_LOW_PCT,
        vix_high_pct: float = VIX_HIGH_PCT,
        tilt_fraction: float = TILT_FRACTION,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            vix_smooth=vix_smooth,
            vix_pct_window=vix_pct_window,
            vix_low_pct=vix_low_pct,
            vix_high_pct=vix_high_pct,
            tilt_fraction=tilt_fraction,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.vix_smooth = int(vix_smooth)
        self.vix_pct_window = int(vix_pct_window)
        self.vix_low_pct = float(vix_low_pct)
        self.vix_high_pct = float(vix_high_pct)
        self.tilt_fraction = float(tilt_fraction)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vol_window, self.vix_pct_window) + 10
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

        # --- VIX percentile rank signal ---
        vix_regime = "neutral"
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= self.vix_pct_window + self.vix_smooth:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= self.vix_pct_window + self.vix_smooth:
                    # Smoothed VIX with 10d MA
                    vix_smoothed = vix_close.rolling(self.vix_smooth).mean().dropna()
                    if len(vix_smoothed) >= self.vix_pct_window:
                        window = vix_smoothed.iloc[-self.vix_pct_window:].values
                        current = float(vix_smoothed.iloc[-1])
                        pct = float(np.mean(window < current))
                        if pct <= self.vix_low_pct:
                            vix_regime = "low"
                        elif pct >= self.vix_high_pct:
                            vix_regime = "high"
        except Exception:
            pass

        # --- Compute inverse-vol base weights ---
        base_assets = ["SPY", "IEF", "GLD"]
        prices = ctx.closes_window(self.vol_window + 5)

        inv_vols: dict[str, float] = {}
        for sym in base_assets:
            if sym not in prices.columns or sym not in live:
                continue
            col = prices[sym].dropna()
            if len(col) < self.vol_window + 1:
                continue
            daily_rets = col.pct_change().dropna()
            if len(daily_rets) < 5:
                continue
            vol = float(daily_rets.iloc[-self.vol_window:].std())
            if vol > 0:
                inv_vols[sym] = 1.0 / vol

        if not inv_vols:
            return []

        total_iv = sum(inv_vols.values())
        base_w = {s: inv_vols[s] / total_iv for s in inv_vols}

        spy_w = base_w.get("SPY", 0.0)
        ief_w = base_w.get("IEF", 0.0)
        gld_w = base_w.get("GLD", 0.0)

        final: dict[str, float] = {}
        if vix_regime == "low":
            # Low-vol regime: growth tilt — shift 15% of IEF into SPY
            tilt = ief_w * self.tilt_fraction
            final["SPY"] = spy_w + tilt
            final["IEF"] = ief_w - tilt
            final["GLD"] = gld_w
        elif vix_regime == "high":
            # High-vol regime: defensive tilt — shift 15% of SPY into GLD
            tilt = spy_w * self.tilt_fraction
            final["SPY"] = spy_w - tilt
            final["IEF"] = ief_w
            final["GLD"] = gld_w + tilt
        else:
            final.update(base_w)

        target = {
            s: w * self.exposure
            for s, w in final.items()
            if s in live and w > 0
        }

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
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


UNIVERSE = ["SPY", "IEF", "GLD", "^VIX"]

NAME = "opus1_rp_vix_regime_tilt"
HYPOTHESIS = (
    "Mutation of rp_yield_curve_tilt: same SPY/IEF/GLD inverse-vol base; replace 10Y-2Y "
    "curve slope tilt with VIX 10d MA percentile rank tilt (bottom quartile shift IEF->SPY, "
    "top quartile shift SPY->GLD); 15% tilt fraction; monthly rebalance; goal more stable "
    "h1/h2 via self-normalizing signal"
)

STRATEGY = RpVixRegimeTilt()
