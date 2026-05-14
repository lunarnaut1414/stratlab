"""Yield-curve-slope cross-factor rotation — gen_7 sonnet-8

Hypothesis: Use the 10Y-2Y Treasury yield spread (^TNX minus ^IRX) as a macro
regime signal to rotate between growth equities, blended equity/bonds, and a
defensive macro allocation.

- Steep curve  (spread >= 1.5%): hold QQQ 97% (growth outperforms in reflation)
- Moderate curve (0% to 1.5%): hold SPY 60% + IEF 37% (balanced; modest growth)
- Inverted/flat  (spread < 0%): hold TLT 60% + GLD 37% (deflation/recession hedge)

Weekly rebalance. Use 20d smoothing on the yield spread to reduce whipsaws.

Rationale: The yield curve has a well-documented leading relationship with the
economic cycle. Steep = healthy expansion (risk-on growth), flat = late cycle
(blend), inverted = recession signal (defensive). This is structurally different
from VIX-level, credit-spread, and breadth-based regime signals on the leaderboard.
^TNX and ^IRX are signal-only (not tradeable); QQQ/SPY/IEF/TLT/GLD are the vehicles.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
SMOOTH_DAYS = 20         # 20d MA on spread to reduce noise
STEEP_THRESHOLD = 1.5    # spread >= this -> growth (QQQ)
FLAT_THRESHOLD = 0.0     # spread < this -> defensive (TLT+GLD)
EXPOSURE = 0.97

_TNX = "^TNX"  # 10-year yield
_IRX = "^IRX"  # 13-week (≈2y proxy) yield


class YieldCurveSlopeRotation(Strategy):
    """Yield-curve-slope regime rotation: QQQ (steep) / SPY+IEF (moderate) / TLT+GLD (inverted)."""

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

        # Read yield curve signals
        spread_smooth = float("nan")
        try:
            tnx_hist = ctx.history(_TNX)
            irx_hist = ctx.history(_IRX)
            if len(tnx_hist) >= self.smooth_days + 5 and len(irx_hist) >= self.smooth_days + 5:
                tnx_close = tnx_hist["close"].dropna()
                irx_close = irx_hist["close"].dropna()
                # Align by taking last N points
                n = min(len(tnx_close), len(irx_close), self.smooth_days + 5)
                tnx_arr = tnx_close.values[-n:]
                irx_arr = irx_close.values[-n:]
                # Compute spread (^TNX is in percent, ^IRX is annualized discount rate in percent)
                # Both expressed as percentages so spread = TNX - IRX
                spread_arr = tnx_arr - irx_arr
                if len(spread_arr) >= self.smooth_days:
                    spread_smooth = float(np.mean(spread_arr[-self.smooth_days:]))
        except Exception:
            pass

        # Determine regime
        # If signal unavailable, default to moderate (balanced) allocation
        if np.isnan(spread_smooth):
            target = {"SPY": 0.60 * self.exposure, "IEF": 0.37 * self.exposure}
        elif spread_smooth >= self.steep_threshold:
            # Steep curve: growth regime — QQQ
            target = {"QQQ": self.exposure}
        elif spread_smooth < self.flat_threshold:
            # Inverted/flat: defensive — TLT + GLD
            target = {"TLT": 0.60 * self.exposure, "GLD": 0.37 * self.exposure}
        else:
            # Moderate: blend — SPY + IEF
            target = {"SPY": 0.60 * self.exposure, "IEF": 0.37 * self.exposure}

        # Build orders
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


NAME = "yield_curve_slope_rotation"
HYPOTHESIS = (
    "Yield-curve-slope cross-factor rotation: when 10Y-2Y spread (TNX-IRX) above 1.5% "
    "hold QQQ 97%; when 0-1.5% hold SPY 60%+IEF 37%; when inverted (<0%) hold TLT 60%+GLD 37%; "
    "weekly rebalance; distinct signal from existing VIX/credit/breadth regime gates"
)
UNIVERSE = ["QQQ", "SPY", "IEF", "TLT", "GLD", _TNX, _IRX]

STRATEGY = YieldCurveSlopeRotation()
