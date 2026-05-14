"""Sector ETF Momentum Rotation (Top-5 Sectors) — gen_8 sonnet-4

Hypothesis: When SPY is above its 200d SMA (bull market), rank the 9 SPDR
sector ETFs (XLK, XLF, XLV, XLI, XLE, XLB, XLY, XLP, XLU) by 63d return
and hold the top-5 equally weighted. When SPY is below 200d SMA (bear market)
rotate to a defensive mix of XLU 40%, XLP 30%, TLT 27% (the three most
defensive assets: utilities, staples, bonds). Weekly rebalance.

Rationale: Pure sector rotation with market-trend gate. Holding 5 of 9 sectors
(not just 2-3) gives enough diversification that the portfolio isn't too
concentrated in single sectors. The defensive bucket uses XLU+XLP instead of
just TLT, providing inflation-resistant equity defensives that outperformed
plain bonds in the 2013-2018 tightening cycle.

Distinction from existing strategies:
- gen7_sector_breadth_momentum (dead end): used breadth COUNT as gate signal,
  only held top-2/3 sectors (too concentrated, IS Calmar 0.032)
- gen5_dual_momentum_sector_etf (dead end): used "beat SPY" absolute filter
  which filtered to cash too often
- This uses SPY 200d SMA as market gate (not breadth count), holds TOP-5
  of 9 sectors (more diversified than top-2/3), defensive mix includes
  XLU+XLP not just TLT.
- No SP500 individual stock selection — pure ETF allocator.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
MOMENTUM_WINDOW = 63       # ~3 months sector momentum
TREND_WINDOW = 200         # SPY market gate
TOP_K = 5                  # hold top-5 of 9 sectors
EXPOSURE = 0.97

# Available sector ETFs with full IS window coverage (no XLC - started 2018; no XLRE - started 2015)
_SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLE", "XLB", "XLY", "XLP", "XLU"]

_DEFENSIVE = {
    "XLU": 0.40,   # utilities
    "XLP": 0.30,   # consumer staples
    "TLT": 0.27,   # bonds
}


class SectorTop5Momentum(Strategy):
    """Top-5 sector ETF rotation with SPY 200d bear gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA market gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: XLU + XLP + TLT
            for sym, w in _DEFENSIVE.items():
                if sym in live:
                    target[sym] = w * self.exposure
        else:
            # Rank sectors by 63d momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            for sym in _SECTORS:
                if sym not in prices.columns or sym not in live:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 1:
                    continue
                p_start = float(col.iloc[-self.momentum_window])
                p_end = float(col.iloc[-1])
                if p_start <= 0:
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 3:
                # Not enough sector data — fallback to SPY
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

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


UNIVERSE = _SECTORS + ["SPY", "TLT"]

NAME = "sector_top5_momentum"
HYPOTHESIS = (
    "Sector ETF momentum rotation: when SPY above 200d SMA, rank XLK/XLF/XLV/XLI/XLE/XLB/XLY/XLP/XLU "
    "by 63d return, hold top-5 equally weighted; when SPY below 200d SMA hold XLU 40%+XLP 30%+TLT 27% "
    "defensive mix; weekly rebalance; distinct from single-stock SP500 momentum and ETF cross-asset allocators"
)

STRATEGY = SectorTop5Momentum()
