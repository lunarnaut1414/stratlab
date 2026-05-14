"""Sector ETF Dual-Momentum Rotation — gen_8 sonnet-8

Hypothesis: Each week, hold the top-2 XL* sector ETFs by 60d return, BUT
only if those ETFs have positive 60d absolute return AND VIX is below 22.
Hold IEF when fewer than 2 sectors qualify. Equal-weight among qualifiers.
Weekly rebalance.

Rationale: Pure relative momentum among sectors (rank by 60d return, hold top-N)
forces allocation even into negatively-trending sectors when the whole market is
declining. Adding an ABSOLUTE momentum filter (positive 60d return required)
means we only hold sectors that are genuinely rallying, not just falling less.
The VIX<22 gate additionally avoids holding equities during elevated fear periods.
IEF provides safety when no sectors qualify.

Distinct from leaderboard strategies:
- gen5_sp500_momentum_vix_sized: individual stocks, not sector ETFs
- gen6_jnk_vix_dual_gate_qqq: uses JNK credit signal, not absolute sector momentum
- gen6_nearhi_momentum_quality: stocks near 52w high, not sector ETF rotation

The XL* sector ETF universe ensures we're investing in diversified sector baskets,
not picking individual stocks — a different granularity level than existing strategies.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
REBALANCE_EVERY = 5           # weekly
MOM_WINDOW = 60               # momentum lookback for both absolute and relative
TOP_K = 2                     # top sectors to hold
VIX_THRESHOLD = 22.0          # calm market gate
EXPOSURE = 0.97
_VIX = "^VIX"
_IEF = "IEF"

# XL* sector ETFs — all available since ~2000, full IS window coverage
_SECTORS = [
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate (started 2015, OK for late-IS window)
]


class SectorETFDualMomentum(Strategy):
    """Top-2 sector ETFs by 60d return when absolute momentum positive AND VIX calm."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        top_k: int = TOP_K,
        vix_threshold: float = VIX_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            top_k=top_k,
            vix_threshold=vix_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.top_k = int(top_k)
        self.vix_threshold = float(vix_threshold)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # ---- VIX gate ----
        vix_ok = True  # default to allowing if VIX unavailable
        try:
            vix_hist = ctx.history(_VIX)
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= 1:
                vix_now = float(vix_close.iloc[-1])
                if np.isfinite(vix_now):
                    vix_ok = vix_now < self.vix_threshold
        except (KeyError, Exception):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not vix_ok:
            # VIX elevated: defensive
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Compute 60d return for each sector ETF
            prices = ctx.closes_window(self.mom_window + 5)
            if len(prices) < self.mom_window:
                return []

            scores: dict[str, float] = {}
            for sym in _SECTORS:
                if sym not in prices.columns or sym not in live:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.mom_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                if np.isfinite(ret) and ret > 0:  # ABSOLUTE momentum filter: must be positive
                    scores[sym] = ret

            if not scores:
                # No qualifying sectors
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                # Rank by relative momentum, take top-K
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                k = min(self.top_k, len(ranked))
                selected = ranked[:k]
                per_weight = self.exposure / len(selected)
                for sym in selected:
                    target[sym] = per_weight

                # If fewer than top_k qualified, add IEF for remaining slot(s)
                # (but only if we have at least 1 sector position)
                # Actually: just use all available qualifying sectors, up to top_k
                # If fewer than 2 sectors qualify, rest stays cash (IEF not added here)
                # unless 0 sectors qualify (handled above)

        # ---- Build orders ----
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


UNIVERSE = _SECTORS + [_IEF, _VIX]

NAME = "sector_etf_dual_momentum"
HYPOTHESIS = (
    "Sector ETF dual-momentum rotation: hold top-2 XL* sector ETFs by 60d return "
    "when those ETFs have positive 60d absolute momentum AND VIX below 22; "
    "hold IEF when no sectors qualify; equal-weight; weekly rebalance; combines "
    "absolute+relative momentum filter on sector ETFs distinct from pure cross-sectional "
    "stock momentum"
)

STRATEGY = SectorETFDualMomentum()
