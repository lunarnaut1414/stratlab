"""Risk Parity with Yield Curve Regime Tilt — gen_8 sonnet-4

Hypothesis: Baseline inverse-vol weighted risk parity on SPY/IEF/GLD (always
invested). Apply a yield-curve regime tilt using the 10Y-2Y spread (TNX-IRX):
  - When curve steep (TNX - IRX > 1.5%): shift 15% of IEF allocation to SPY
    (growth regime — steep curve predicts expansion, favor equity over bonds)
  - When curve flat/inverted (TNX - IRX < 0%): shift 15% of SPY allocation to GLD
    (stagflation hedge — inverted curve predicts contraction + inflation risk)
  - Otherwise (0% to 1.5%): plain inverse-vol RP weights

Monthly rebalance. TNX and IRX are signal-only (not tradeable).

Rationale: Yield curve slope is one of the most empirically robust macro signals.
In 2010-2018:
  - 2010-2014: steep curve (accommodative Fed, growth) → equity tilt adds alpha
  - 2015-2018: flattening (Fed hike cycle) → neutral/GLD tilt prevents bonds-only drag
  - Always-invested RP baseline ensures no dead time in cash

Distinction from existing strategies:
  - gen6_bond_termstruct_curve_rotation: pure bond-only duration rotation (no equity)
  - gen7_yield_curve_slope_rotation: discrete 3-tier ETF switching (QQQ/SPY+IEF/TLT+GLD)
  - gen7_opus1_longend_curve_mtum_rotation: long-end (30Y-10Y) curve with MTUM
  - This uses 10Y-2Y slope as a TILT on top of RP baseline, keeping always-invested
    with smoothly varying exposures. Different from discrete binary switches.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
VOL_WINDOW = 20            # for inverse-vol sizing
STEEP_THRESHOLD = 1.5      # 10Y-2Y spread threshold for growth tilt (%)
FLAT_THRESHOLD = 0.0       # 10Y-2Y spread threshold for stagflation tilt (%)
CURVE_TILT = 0.15          # fraction to shift in each regime
SMOOTH_DAYS = 10           # MA smoothing for yield spread noise
EXPOSURE = 0.97


class RpYieldCurveTilt(Strategy):
    """Risk parity SPY/IEF/GLD with 10Y-2Y yield curve slope tilt."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        steep_threshold: float = STEEP_THRESHOLD,
        flat_threshold: float = FLAT_THRESHOLD,
        curve_tilt: float = CURVE_TILT,
        smooth_days: int = SMOOTH_DAYS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            steep_threshold=steep_threshold,
            flat_threshold=flat_threshold,
            curve_tilt=curve_tilt,
            smooth_days=smooth_days,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.steep_threshold = float(steep_threshold)
        self.flat_threshold = float(flat_threshold)
        self.curve_tilt = float(curve_tilt)
        self.smooth_days = int(smooth_days)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vol_window, self.smooth_days) + 10
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

        # Compute 10Y-2Y yield spread (signal-only: ^TNX and ^IRX)
        curve_slope = 1.0  # default neutral (slight steepness)
        try:
            tnx_hist = ctx.history("^TNX")
            irx_hist = ctx.history("^IRX")
            if (tnx_hist is not None and len(tnx_hist) >= self.smooth_days + 2 and
                    irx_hist is not None and len(irx_hist) >= self.smooth_days + 2):
                tnx_close = tnx_hist["close"].dropna()
                irx_close = irx_hist["close"].dropna()
                if len(tnx_close) >= self.smooth_days and len(irx_close) >= self.smooth_days:
                    # Smooth over last smooth_days to reduce noise
                    tnx_smooth = float(tnx_close.iloc[-self.smooth_days:].mean())
                    irx_smooth = float(irx_close.iloc[-self.smooth_days:].mean())
                    curve_slope = tnx_smooth - irx_smooth
        except Exception:
            pass

        # Compute inverse-vol for base assets
        base_assets = ["SPY", "IEF", "GLD"]
        need = self.vol_window + 5
        prices = ctx.closes_window(need)

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

        total_inv_vol = sum(inv_vols.values())
        base_weights = {sym: inv_vols[sym] / total_inv_vol for sym in inv_vols}

        spy_w = base_weights.get("SPY", 0.0)
        ief_w = base_weights.get("IEF", 0.0)
        gld_w = base_weights.get("GLD", 0.0)

        # Apply yield curve tilt
        final_weights: dict[str, float] = {}
        if curve_slope > self.steep_threshold:
            # Steep curve: growth regime — shift 15% of IEF to SPY
            tilt = ief_w * self.curve_tilt
            final_weights["SPY"] = spy_w + tilt
            final_weights["IEF"] = ief_w - tilt
            final_weights["GLD"] = gld_w
        elif curve_slope < self.flat_threshold:
            # Flat/inverted curve: stagflation — shift 15% of SPY to GLD
            tilt = spy_w * self.curve_tilt
            final_weights["SPY"] = spy_w - tilt
            final_weights["IEF"] = ief_w
            final_weights["GLD"] = gld_w + tilt
        else:
            # Neutral: plain risk parity
            final_weights.update(base_weights)

        # Apply exposure cap
        target: dict[str, float] = {
            sym: w * self.exposure
            for sym, w in final_weights.items()
            if sym in live and w > 0
        }

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
    return ["SPY", "IEF", "GLD", "TLT", "^TNX", "^IRX"]


UNIVERSE = _universe

NAME = "rp_yield_curve_tilt"
HYPOTHESIS = (
    "Risk parity SPY/IEF/GLD with yield curve regime tilt: baseline inverse-vol RP on "
    "SPY/IEF/GLD; when 10Y-2Y yield curve steep (TNX-IRX > 1.5%) increase SPY allocation "
    "by 15% of IEF weight (growth tilt); when inverted (< 0%) increase GLD by 15% of SPY "
    "weight (stagflation tilt); monthly rebalance; yield-curve tilted RP distinct from "
    "JNK credit tilt variant"
)

STRATEGY = RpYieldCurveTilt()
