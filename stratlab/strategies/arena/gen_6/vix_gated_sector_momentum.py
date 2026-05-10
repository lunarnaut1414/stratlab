"""VIX-gated SPDR sector ETF top-2 momentum — gen_6 sonnet-7

Hypothesis: When VIX < 20 (calm market), hold top-2 SPDR sector ETFs
(XLK, XLV, XLF, XLI, XLE, XLU, XLB, XLP) by 42-day momentum. When
VIX >= 20, rotate to TLT + GLD equally (40% each). Rebalance every 10 bars.

Distinct from existing strategies:
  - sector ETFs only (no individual stocks)
  - VIX < 20 threshold (different from VIX < 25 used in gen5_vix_gated_sp500_momentum)
  - TLT + GLD as defensive (not SHY+TLT or IEF alone)
  - 42-day momentum window (different from 63d/21d used elsewhere)
  - top-2 sectors (not top-3 or all sectors)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Core SPDR sector ETFs (all active since 1998)
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLI", "XLE", "XLU", "XLB", "XLP"]

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 42    # 42-day momentum
VIX_THRESHOLD = 20.0    # VIX<20 = calm (stricter than 25)
TOP_K = 2               # top-2 sectors
EXPOSURE = 0.97

_VIX = "^VIX"


class VIXGatedSectorMomentum(Strategy):
    """VIX<20-gated SPDR sector top-2 momentum: TLT+GLD when VIX elevated."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vix_threshold: float = VIX_THRESHOLD,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vix_threshold=vix_threshold,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vix_threshold = float(vix_threshold)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # VIX gate
        vix_calm = True
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= 1:
                vl = float(vix_hist["close"].iloc[-1])
                if np.isfinite(vl):
                    vix_calm = vl < self.vix_threshold
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not vix_calm:
            # VIX elevated: TLT 40% + GLD 40%
            def_etfs = ["TLT", "GLD"]
            avail_def = [s for s in def_etfs if s in closes_now.index]
            if avail_def:
                w = 0.40 * self.exposure
                for sym in avail_def:
                    target[sym] = w
        else:
            # VIX calm: top-2 sector ETFs by 42d momentum
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in SECTOR_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                return []

            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
            per_weight = self.exposure / k
            for sym in ranked:
                target[sym] = per_weight

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


NAME = "vix_gated_sector_momentum"
HYPOTHESIS = (
    "VIX<20 gated SPDR sector top-2 momentum: hold top-2 of XLK/XLV/XLF/XLI/XLE/XLU/XLB/XLP "
    "by 42d return when VIX<20; TLT 40%+GLD 40% when VIX>=20; biweekly rebalance; "
    "VIX<20 threshold (stricter than VIX<25) + TLT+GLD defensive blend"
)
UNIVERSE = SECTOR_ETFS + ["TLT", "GLD", _VIX]
STRATEGY = VIXGatedSectorMomentum()
