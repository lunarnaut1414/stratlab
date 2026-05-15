"""Sector ETF rotation — top 3 by 63d momentum — gen_10 sonnet-7

Hypothesis: Each month rank the 6 sector ETFs that cover full IS window
(XLK, XLF, XLE, XLI, XLU, XLY, XLB) by 63d total return. Hold the top-3
with positive absolute return, equal-weight. If fewer than 3 have positive
returns, fill remaining slots with TLT. SPY 200d outer bear gate to TLT.
Rebalance monthly.

Rationale:
  - Pure sector rotation (no single stocks) means this is structurally
    different from all SP500 cross-sectional strategies (corr will be low).
  - Using ONLY sector ETFs that cover full IS (7 XL* ETFs) avoids the
    coverage gaps that killed XLV/XLP/XLRE-based strategies.
  - 63d momentum (3-month) in sectors has different persistence than
    individual stock momentum — sectors mean-revert slower.
  - The 3-of-7 selection with TLT fill creates an automatic defensive
    buffer during sector drawdowns (fewer than 3 positive sectors = defensive
    regime). This is mechanism-different from macro gates.
  - This is NOT a macro-signal allocator — it's a pure cross-sectional
    momentum rotation with a structural defensive buffer.

Data checks:
  - XLK, XLF, XLE, XLI, XLU, XLY, XLB: all start 1998, full IS coverage.
  - TLT: starts 2002, full IS coverage.
  - SPY: full IS coverage.
  - No XLP, XLV, XLRE (coverage gaps confirmed).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21     # monthly
MOMENTUM_WINDOW = 63     # 3-month sector momentum
SPY_TREND_WINDOW = 200   # outer bear gate
TOP_K = 3                # top-3 sectors
EXPOSURE = 0.97

SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLU", "XLY", "XLB"]


def _universe() -> list[str]:
    return SECTORS + ["TLT", "SPY"]


UNIVERSE = _universe


class SectorRotationTop3(Strategy):
    """Top-3 sector ETFs by 63d momentum; TLT fill for sub-minimum slots;
    SPY 200d outer bear gate to TLT; monthly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d outer bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Compute 63d momentum for each sector ETF
            scores: dict[str, float] = {}
            for sym in SECTORS:
                if sym not in live:
                    continue
                try:
                    hist = ctx.history(sym)
                except KeyError:
                    continue
                closes = hist["close"].dropna()
                if len(closes) < self.momentum_window + 2:
                    continue
                p_end = float(closes.iloc[-1])
                p_start = float(closes.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            # Sort by momentum descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Take top-K with positive absolute return
            selected_sectors = [sym for sym, ret in ranked[:self.top_k] if ret > 0]
            n_tlt_fill = self.top_k - len(selected_sectors)  # fill remaining with TLT

            total_slots = self.top_k
            per_slot_w = self.exposure / total_slots

            for sym in selected_sectors:
                target[sym] = per_slot_w

            if n_tlt_fill > 0 and "TLT" in live:
                tlt_w = per_slot_w * n_tlt_fill
                # Add to existing TLT weight if any
                target["TLT"] = target.get("TLT", 0.0) + tlt_w

            # If no sectors at all with data, hold TLT
            if not target and "TLT" in live:
                target["TLT"] = self.exposure

        # Build orders
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


NAME = "sector_rotation_top3"
HYPOTHESIS = (
    "Sector ETF rotation: rank XLK/XLF/XLE/XLI/XLU/XLY/XLB by 63d return monthly; hold "
    "top-3 with positive absolute return equal-weight; fill remaining slots with TLT; "
    "SPY 200d bear gate to TLT; monthly rebalance — pure sector cross-sectional momentum "
    "with structural defensive buffer, no single stocks"
)

STRATEGY = SectorRotationTop3()
