"""Long-end (30Y-10Y) yield-curve-segment rotation — opus-1 gen_7

Mutation of gen7_yield_curve_slope_rotation (parent IS Calmar 0.82).

Parent: 10Y-2Y short-end spread (TNX-IRX); QQQ / SPY+IEF / TLT+GLD.
Mutation: replace the curve segment with the LONG END (30Y-10Y, ^TYX-^TNX).
The long-end spread captures different macro information: term-premium
expectations and long-run inflation fears, vs the short-end which is more
about Fed-policy expectations. A steepening long end signals expansion-driven
re-pricing of inflation; a flattening long end is more often associated with
stagflation regimes.

Also change the risk-on allocation: MTUM (factor-momentum ETF) instead of
QQQ. MTUM has different sector exposures than QQQ (less tech-concentrated,
more momentum-rotational), which should produce a meaningfully different
return profile in long-end-steep regimes.

Thresholds tuned for the 30Y-10Y spread distribution (smaller magnitude than
TNX-IRX historically — typical range 0% to +1%):
  - spread_smooth >=  0.5%  : steep long end → MTUM 97%
  - 0.5% > spread > -0.2%   : moderate         → SPY 60% + IEF 37%
  - spread <= -0.2%         : inverted long end → TLT 60% + GLD 37%

Different curve segment + different risk-on vehicle = distinct strategy.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
SMOOTH_DAYS = 20
STEEP_THRESHOLD = 0.5
FLAT_THRESHOLD = -0.2
EXPOSURE = 0.97

_TYX = "^TYX"  # 30-year yield
_TNX = "^TNX"  # 10-year yield


class LongEndCurveMtumRotation(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        smooth_days: int = SMOOTH_DAYS,
        steep_threshold: float = STEEP_THRESHOLD,
        flat_threshold: float = FLAT_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            smooth_days=smooth_days,
            steep_threshold=steep_threshold,
            flat_threshold=flat_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.smooth_days = int(smooth_days)
        self.steep_threshold = float(steep_threshold)
        self.flat_threshold = float(flat_threshold)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.smooth_days + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute long-end spread (30Y - 10Y), smoothed 20d
        spread_smooth = float("nan")
        try:
            tyx_hist = ctx.history(_TYX)
            tnx_hist = ctx.history(_TNX)
            if len(tyx_hist) >= self.smooth_days + 5 and len(tnx_hist) >= self.smooth_days + 5:
                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()
                n = min(len(tyx_close), len(tnx_close), self.smooth_days + 5)
                tyx_arr = tyx_close.values[-n:]
                tnx_arr = tnx_close.values[-n:]
                spread_arr = tyx_arr - tnx_arr
                if len(spread_arr) >= self.smooth_days:
                    spread_smooth = float(np.mean(spread_arr[-self.smooth_days:]))
        except Exception:
            pass

        # Decide regime
        if np.isnan(spread_smooth):
            target = {"SPY": 0.60 * self.exposure, "IEF": 0.37 * self.exposure}
        elif spread_smooth >= self.steep_threshold:
            # Steep long end → momentum factor (MTUM)
            target = {"MTUM": self.exposure}
            # Fallback to QQQ if MTUM not available (pre-2013 inception risk)
            if "MTUM" not in live:
                target = {"QQQ": self.exposure}
                if "QQQ" not in live:
                    target = {"SPY": self.exposure}
        elif spread_smooth <= self.flat_threshold:
            target = {"TLT": 0.60 * self.exposure, "GLD": 0.37 * self.exposure}
        else:
            target = {"SPY": 0.60 * self.exposure, "IEF": 0.37 * self.exposure}

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
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


NAME = "opus1_longend_curve_mtum_rotation"
HYPOTHESIS = (
    "Curve-segment rotation: use 30Y-10Y spread (TYX minus TNX) as long-end curve "
    "signal instead of 10Y-2Y; steep long-end (>0.5%) = MTUM (momentum factor ETF) "
    "not QQQ; flat (-0.2 to 0.5) = SPY+IEF blend; inverted long-end (<-0.2) = TLT+GLD; "
    "long-end signal has different regime profile from short-end TNX-IRX"
)
UNIVERSE = ["MTUM", "QQQ", "SPY", "IEF", "TLT", "GLD", _TYX, _TNX]

STRATEGY = LongEndCurveMtumRotation()
