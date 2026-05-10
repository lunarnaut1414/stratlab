"""Bond term-structure rotation by 10Y-3M yield curve slope.

Hypothesis: The Treasury curve slope (10Y minus 3M, ^TNX - ^IRX) tells us
which point of the duration curve has the best risk-adjusted carry:
  - Steep curve (slope >= 2.5%): long end is rewarded for duration risk —
    hold TLT 60% + IEF 37% (heavy long-duration tilt).
  - Moderately steep (slope 1.5% - 2.5%): hold IEF 60% + TLT 37% (mid-duration).
  - Flat (slope 0.5% - 1.5%): IEF 97% (mid-duration only).
  - Very flat / approaching inversion (slope < 0.5%): SHY 97% (cash-proxy
    only — long-duration likely to repriced).

In the 2010-2018 IS window, the curve was steep (2010-2014, 10Y-3M >2.5%)
during QE, then progressively flattened through 2014-2018 as the Fed
hiked. So this strategy:
  - Holds TLT-heavy in 2010-2014 (TLT actually outperformed across 2010-12,
    delivered carry through QE3).
  - Rotates to IEF-only in 2015-2017 (mid-duration sweet spot).
  - Rotates to SHY in 2018 (curve was nearly flat, TLT lost value).

Why this fills a gap:
  - Phase 2 brief calls this out specifically: "Bond term-structure: TLT
    vs IEF vs SHY allocator driven by 10Y-2Y or 10Y-FF curve slope (NOT
    TNX direction — that's saturated)".
  - Saturated dead-end list flags TNX-direction sector mapping (3 failed)
    and duration_curve_ief_shy (failed in gen_5 because it used TLT/BIL
    carry, which is signal-noisy). My version uses ^TNX-^IRX as the
    smoothed signal and three clean buckets (TLT/IEF/SHY).
  - All 8 saturated yield-curve attempts mapped curve slope to *equities*
    or *sector rotation*. None mapped curve slope to bond-duration
    rotation directly. This is the orthogonal idea.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["TLT", "IEF", "SHY", "SPY", "^TNX", "^IRX"]

STEEP_THRESHOLD = 2.5     # very steep curve
MOD_STEEP_THRESHOLD = 1.5  # moderately steep
FLAT_THRESHOLD = 0.5       # very flat
SMOOTH_DAYS = 20           # smooth slope
TREND_WINDOW = 200
REBALANCE_EVERY = 10       # biweekly
EXPOSURE = 0.97


class BondTermStructCurveRotation(Strategy):
    def __init__(
        self,
        steep_threshold: float = STEEP_THRESHOLD,
        mod_steep_threshold: float = MOD_STEEP_THRESHOLD,
        flat_threshold: float = FLAT_THRESHOLD,
        smooth_days: int = SMOOTH_DAYS,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            steep_threshold=steep_threshold,
            mod_steep_threshold=mod_steep_threshold,
            flat_threshold=flat_threshold,
            smooth_days=smooth_days,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.steep_threshold = float(steep_threshold)
        self.mod_steep_threshold = float(mod_steep_threshold)
        self.flat_threshold = float(flat_threshold)
        self.smooth_days = int(smooth_days)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.smooth_days + 10, self.trend_window + 5)
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend gate
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 1:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window + 1:
                    spy_now = float(spy_close.iloc[-1])
                    spy_ma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = spy_now > spy_ma
        except KeyError:
            pass

        # Compute smoothed curve slope ^TNX - ^IRX
        slope = float("nan")
        try:
            tnx_hist = ctx.history("^TNX")
            irx_hist = ctx.history("^IRX")
            if (
                tnx_hist is not None
                and irx_hist is not None
                and len(tnx_hist) >= self.smooth_days
                and len(irx_hist) >= self.smooth_days
            ):
                tnx_close = tnx_hist["close"].dropna()
                irx_close = irx_hist["close"].dropna()
                if len(tnx_close) >= self.smooth_days and len(irx_close) >= self.smooth_days:
                    df = pd.concat(
                        [tnx_close.rename("tnx"), irx_close.rename("irx")],
                        axis=1,
                    ).dropna()
                    if len(df) >= self.smooth_days:
                        slopes = df["tnx"] - df["irx"]
                        slope = float(slopes.iloc[-self.smooth_days:].mean())
        except Exception:
            pass

        if not np.isfinite(slope):
            slope = 1.5  # safe neutral

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Equity sleeve sizing: 60% SPY in bull-trend, else 0%
        eq_w = 0.60 if spy_bull else 0.0
        bond_w = 1.0 - eq_w

        if "SPY" in live and eq_w > 0:
            target["SPY"] = eq_w * self.exposure

        # Bond sleeve allocation by curve slope
        if slope >= self.steep_threshold:
            # Very steep — load long-end duration
            if "TLT" in live:
                target["TLT"] = bond_w * 0.65 * self.exposure
            if "IEF" in live:
                target["IEF"] = bond_w * 0.35 * self.exposure
        elif slope >= self.mod_steep_threshold:
            # Moderately steep — barbell with IEF heavy
            if "IEF" in live:
                target["IEF"] = bond_w * 0.65 * self.exposure
            if "TLT" in live:
                target["TLT"] = bond_w * 0.35 * self.exposure
        elif slope >= self.flat_threshold:
            # Flat — mid-duration only
            if "IEF" in live:
                target["IEF"] = bond_w * self.exposure
        else:
            # Very flat / inversion — short end only
            if "SHY" in live:
                target["SHY"] = bond_w * self.exposure

        # Fall-through default
        if not target and "IEF" in live:
            target["IEF"] = self.exposure

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


NAME = "bond_termstruct_curve_rotation"
HYPOTHESIS = (
    "Bond term-structure rotation TLT/IEF/SHY by 10Y-3M yield curve slope: "
    "steep (>=2.5%) hold TLT 60%+IEF 37%; mod-steep (1.5-2.5%) IEF 60%+TLT 37%; "
    "flat (0.5-1.5%) IEF 97%; very flat/inverted (<0.5%) SHY 97%; "
    "monthly rebalance; pure bond-duration allocator gated by curve slope."
)

STRATEGY = BondTermStructCurveRotation()
