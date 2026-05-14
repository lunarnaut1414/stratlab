"""Sector Breadth Thrust SP500 Rotation — gen_8 sonnet-10

Hypothesis: Use the breadth of sector leadership (count of SPDR sector ETFs
above their 50d SMA) as a macro regime signal, then route to top SP500 stocks
or SPY or TLT based on breadth level.

Regime classification:
  - Broad bull (>=8 of 11 sectors above 50d SMA): hold top-15 SP500 stocks by 63d momentum
  - Mixed (4-7 sectors above 50d SMA): hold SPY 97% (market exposure without concentration)
  - Defensive (<4 sectors above 50d SMA): rotate to TLT 97%

Rationale: The breadth of sector participation (not a single sector or index level)
is a robust leading indicator of market health. When most sectors participate, the
bull run is broad-based and likely to persist. When fewer sectors lead, concentration
risk rises and the market is more fragile. This is distinct from VIX gates, credit
gates, yield curve gates, and raw-momentum selection.

Biweekly rebalance: 10 bars.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # ~3 months for stock selection
SECTOR_SMA = 50           # 50d SMA for sector breadth
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"

# All 11 SPDR sector ETFs — used for breadth signal (tradeable but used as signals here)
_SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLE", "XLI",
    "XLU", "XLY", "XLP", "XLB", "XLRE", "XLC",
]

# Bull threshold: breadth >= 8 of 11
_BULL_BREADTH = 8
# Defensive threshold: breadth < 4 of 11
_DEFENSIVE_BREADTH = 4


class SectorBreadthSP500Rotation(Strategy):
    """Sector breadth-gated SP500 momentum rotation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        sector_sma: int = SECTOR_SMA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        bull_breadth: int = _BULL_BREADTH,
        defensive_breadth: int = _DEFENSIVE_BREADTH,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            sector_sma=sector_sma,
            top_k=top_k,
            exposure=exposure,
            bull_breadth=bull_breadth,
            defensive_breadth=defensive_breadth,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.sector_sma = int(sector_sma)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.bull_breadth = int(bull_breadth)
        self.defensive_breadth = int(defensive_breadth)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sector_sma + self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Compute sector breadth: how many sectors above their 50d SMA
        sectors_above = 0
        sectors_counted = 0
        for sector in _SECTOR_ETFS:
            try:
                hist = ctx.history(sector)
            except (KeyError, Exception):
                continue
            if len(hist) < self.sector_sma + 2:
                continue
            close_col = hist["close"].dropna()
            if len(close_col) < self.sector_sma:
                continue
            sma = float(close_col.iloc[-self.sector_sma:].mean())
            now = float(close_col.iloc[-1])
            sectors_counted += 1
            if now > sma:
                sectors_above += 1

        # Fallback: if fewer than 6 sectors counted, use SPY default
        if sectors_counted < 6:
            regime = "mixed"
        elif sectors_above >= self.bull_breadth:
            regime = "bull"
        elif sectors_above < self.defensive_breadth:
            regime = "defensive"
        else:
            regime = "mixed"

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if regime == "defensive":
            if _TLT in live:
                target[_TLT] = self.exposure

        elif regime == "mixed":
            if _SPY in live:
                target[_SPY] = self.exposure

        else:  # bull — pick top-K SP500 stocks by momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                # Fall back to SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT) or sym in _SECTOR_ETFS:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_TLT, _SPY] + _SECTOR_ETFS


NAME = "sector_breadth_sp500_rotation"
HYPOTHESIS = (
    "Sector breadth thrust SP500 momentum: count how many SPDR sector ETFs "
    "(XLK,XLV,XLF,XLE,XLI,XLU,XLY,XLP,XLB,XLRE,XLC) are above their 50d SMA; "
    "when 8+ bullish hold top-15 SP500 stocks by 63d momentum; "
    "when 4-7 hold SPY 97%; when fewer than 4 rotate to TLT; "
    "biweekly rebalance; breadth-of-leadership signal not a single macro indicator"
)

UNIVERSE = _universe

STRATEGY = SectorBreadthSP500Rotation()
