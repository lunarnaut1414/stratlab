"""Sector Breadth + Momentum Hybrid — gen_7 sonnet-3

Hypothesis: Count SPDR sector ETFs above their 50d SMA as a breadth indicator.
When >= 7 of 11 sectors above 50d SMA (strong breadth), hold top-3 sectors by
42d return; when 4-6 sectors above, hold top-2 sectors; when <4 hold TLT.
SPY 200d SMA bear override. Rebalance every 10 bars.

Rationale: Sector breadth (how many sectors are in uptrends) is a proxy for
the health of the bull market. A market where 7+ of 11 sectors are trending
up is a broad rally, more likely to continue. Using only top sectors by
momentum avoids holding laggards. The tiered approach (top-3 vs top-2 vs TLT)
scales risk exposure to breadth regime.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLE", "XLB", "XLRE", "XLY", "XLC"]
BREADTH_HIGH_THRESH = 7  # >= this many sectors in uptrend -> strong
BREADTH_MID_THRESH = 4   # >= this many sectors in uptrend -> moderate
SECTOR_SMA = 50          # sector ETF trend window
MOMENTUM_WINDOW = 42     # 42d return for sector ranking
TREND_WINDOW = 200       # SPY 200d for bear market gate
REBALANCE_EVERY = 10     # bi-weekly
EXPOSURE = 0.97


class SectorBreadthMomentum(Strategy):
    """Sector breadth-gated momentum rotation: top-3/2/defensive based on
    count of SPDR sector ETFs above 50d SMA; SPY 200d bear override.
    """

    def __init__(
        self,
        breadth_high_thresh: int = BREADTH_HIGH_THRESH,
        breadth_mid_thresh: int = BREADTH_MID_THRESH,
        sector_sma: int = SECTOR_SMA,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            breadth_high_thresh=breadth_high_thresh,
            breadth_mid_thresh=breadth_mid_thresh,
            sector_sma=sector_sma,
            momentum_window=momentum_window,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.breadth_high_thresh = int(breadth_high_thresh)
        self.breadth_mid_thresh = int(breadth_mid_thresh)
        self.sector_sma = int(sector_sma)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.sector_sma) + 10
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

        # SPY 200d SMA bear market gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma200 = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma200

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Compute sector breadth (count of sectors above 50d SMA)
            sectors_above_sma = 0
            sector_momentum: dict[str, float] = {}

            for sector in SECTOR_ETFS:
                try:
                    sector_hist = ctx.history(sector)
                except KeyError:
                    continue
                if len(sector_hist) < self.sector_sma + 5:
                    continue
                sector_close = sector_hist["close"].dropna()
                if len(sector_close) < self.sector_sma:
                    continue

                sector_last = float(sector_close.iloc[-1])
                sector_sma = float(sector_close.iloc[-self.sector_sma:].mean())

                if sector_last > sector_sma:
                    sectors_above_sma += 1

                # Compute 42d momentum for ranking
                if len(sector_close) >= self.momentum_window + 2:
                    p_start = float(sector_close.iloc[-self.momentum_window])
                    if p_start > 0 and np.isfinite(p_start) and np.isfinite(sector_last):
                        ret = sector_last / p_start - 1.0
                        if np.isfinite(ret):
                            sector_momentum[sector] = ret

            if sectors_above_sma >= self.breadth_high_thresh:
                # Strong breadth: hold top-3 sectors by 42d return
                top_k = 3
            elif sectors_above_sma >= self.breadth_mid_thresh:
                # Moderate breadth: hold top-2 sectors
                top_k = 2
            else:
                # Weak breadth: defensive TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
                top_k = 0

            if top_k > 0 and sector_momentum:
                ranked = sorted(sector_momentum, key=sector_momentum.__getitem__, reverse=True)
                selected = [s for s in ranked if sector_momentum[s] > 0][:top_k]

                if not selected:
                    # No sectors have positive momentum
                    if "TLT" in closes_now.index:
                        target["TLT"] = self.exposure
                elif len(selected) < top_k:
                    # Fewer than top_k with positive momentum — hold what we have
                    weight = self.exposure / len(selected)
                    for sym in selected:
                        target[sym] = weight
                else:
                    weight = self.exposure / top_k
                    for sym in selected[:top_k]:
                        target[sym] = weight

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


UNIVERSE = (
    SECTOR_ETFS
    + ["TLT", "SPY"]
)

NAME = "sector_breadth_momentum"
HYPOTHESIS = (
    "Sector breadth + momentum hybrid: count SPDR sector ETFs above 50d SMA as breadth indicator; "
    "when >=7 of 11 sectors above 50d SMA (strong breadth) hold top-3 sectors by 42d return; "
    "when 4-6 sectors above hold top-2 sectors; when <4 hold TLT; "
    "SPY 200d SMA bear override; rebalance every 10 bars"
)

STRATEGY = SectorBreadthMomentum()
