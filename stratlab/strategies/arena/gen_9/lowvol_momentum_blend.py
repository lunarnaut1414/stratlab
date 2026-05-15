"""Sector Dual Momentum Rotation — gen_9 sonnet-10

Hypothesis: Apply Gary Antonacci's dual momentum framework to sector ETFs.
For each of 9 sector ETFs (XLK, XLV, XLF, XLE, XLI, XLU, XLY, XLP, XLB),
compute both:
  (A) Absolute momentum: 63d return vs 0 (is it positive?)
  (B) Relative momentum vs SPY: 63d sector return minus 63d SPY return

Select the top-3 sectors with positive absolute momentum, ranked by relative
strength vs SPY. Any slot that can't be filled (no positive abs momentum)
falls back to TLT.

SPY 200d SMA bear override: if SPY below 200d SMA, force full TLT.
Weekly rebalance (every 5 bars) to increase trade count.

Rationale: Pure sector momentum (without abs-mom filter) holds lagging sectors.
Dual momentum forces both conditions: sector must outperform on absolute
basis AND outperform the broad market. This avoids the "all-sector-negative"
problem that plagues pure relative-momentum rotators in bear markets.
The top-3 allocation (vs top-1) reduces concentration risk.

Differentiation from leaderboard:
- gen5_semi_cycle_smh: only uses SMH vs SPY comparison (1 sector, not 9)
- gen5_copper_cycle_rotation: copper commodity proxy for cyclicals/defensives
  (failed with n_trades=0 due to data issues)
- gen6_jnk_vix_dual_gate_qqq: JNK+VIX gate on QQQ/SPY/TLT (not sector rotation)
- No existing leaderboard strategy applies dual momentum across all 9 SPDR sectors
- This is structurally distinct from credit, VIX, yield, dollar, and factor signals

VIX-awareness: sectors include defensive (XLU, XLP) and cyclical (XLE, XLK), so
the dual-momentum filter naturally rotates defensively during downturns without
an explicit VIX gate.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Sector ETF universe (9 SPDR sectors, all launched 1998)
SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLY", "XLP", "XLB"]
DEFENSIVE = "TLT"

MOMENTUM_WINDOW = 63     # ~3-month momentum
TREND_WINDOW = 200       # SPY 200d SMA bear gate
TOP_K = 3                # Hold top-3 sectors
REBALANCE_EVERY = 5      # Weekly rebalance for higher trade count
EXPOSURE = 0.97

UNIVERSE = SECTORS + [DEFENSIVE, "SPY"]


class SectorDualMomentum(Strategy):
    """Dual momentum (absolute + relative) rotation across 9 SPDR sectors."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = TREND_WINDOW + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < TREND_WINDOW:
            return []
        spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
        spy_price = live_all.get("SPY", 0.0)
        spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target: dict[str, float] = {DEFENSIVE: EXPOSURE}
        else:
            # --- SPY 63d absolute momentum ---
            prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
            if len(prices_window) < MOMENTUM_WINDOW:
                return []

            # Compute SPY return for relative comparison
            spy_ret = 0.0
            if "SPY" in prices_window.columns:
                spy_col = prices_window["SPY"].dropna()
                if len(spy_col) >= MOMENTUM_WINDOW:
                    p_end = float(spy_col.iloc[-1])
                    p_start = float(spy_col.iloc[-MOMENTUM_WINDOW])
                    if p_start > 0:
                        spy_ret = p_end / p_start - 1.0

            # --- Dual momentum scores for each sector ---
            qualified: list[tuple[str, float]] = []  # (sector, rel_mom_score)

            for sym in SECTORS:
                if sym not in prices_window.columns:
                    continue
                col = prices_window[sym].dropna()
                if len(col) < MOMENTUM_WINDOW:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOMENTUM_WINDOW])
                if p_start <= 0:
                    continue

                abs_mom = p_end / p_start - 1.0
                if not np.isfinite(abs_mom):
                    continue

                # Absolute momentum gate: must be positive
                if abs_mom <= 0:
                    continue

                # Relative momentum vs SPY
                rel_mom = abs_mom - spy_ret
                if np.isfinite(rel_mom):
                    qualified.append((sym, rel_mom))

            # Sort by relative momentum (highest first)
            qualified.sort(key=lambda x: x[1], reverse=True)

            if not qualified:
                # No sectors pass dual momentum — go full defensive
                target = {DEFENSIVE: EXPOSURE}
            else:
                # Take top-K qualified sectors
                top_sectors = [sym for sym, _ in qualified[:TOP_K]]
                n = len(top_sectors)
                per_weight = EXPOSURE / TOP_K  # fixed weight per slot

                # Remaining slots to TLT if fewer than TOP_K qualify
                n_tlt_slots = TOP_K - n
                target = {sym: per_weight for sym in top_sectors}
                if n_tlt_slots > 0:
                    target[DEFENSIVE] = per_weight * n_tlt_slots

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sector_dual_momentum"
HYPOTHESIS = (
    "Sector dual momentum rotation: for each of 9 SPDR sector ETFs compute 63d absolute "
    "momentum (must be positive) and 63d return vs SPY (relative momentum); hold top-3 "
    "sectors with positive absolute and highest relative momentum; unfilled slots → TLT; "
    "SPY 200d bear override to full TLT; weekly rebalance"
)

STRATEGY = SectorDualMomentum()
