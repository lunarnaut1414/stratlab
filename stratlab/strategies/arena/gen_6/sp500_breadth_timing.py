"""SP500 breadth-thrust timing strategy — gen_6 sonnet-2

Hypothesis:
  Measure market breadth as the percentage of SP500 stocks trading above
  their own 200-day SMA. Use this breadth as a regime signal:
    - Breadth > 55%  (healthy bull):   hold QQQ at 97% exposure
    - Breadth < 40%  (bear market):    hold TLT 50% + SHY 47%
    - 40% <= Breadth <= 55% (neutral): hold SPY at 97% (moderate)
  Rebalance weekly (every 5 bars).

Rationale:
  When a large majority of stocks are in individual uptrends (above 200d SMA),
  the market has broad participation and momentum is likely to persist.
  When breadth collapses (few stocks above 200d SMA), it signals underlying
  weakness even if indices are still elevated — an early warning system.

  This breadth signal is fundamentally different from:
  - VIX (measures implied volatility, not direction)
  - Credit spreads (fixed-income risk signal)
  - Price momentum (individual stock returns)
  - SPY SMA (index level trend, ignores constituent breadth)

  The RSP/SPY breadth proxy (atr_momentum_etf, IS Calmar 0.740) used a
  different method (ratio of ETFs). This computes breadth directly from
  individual stock data — a more precise measure.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

TREND_WINDOW = 200      # SMA window for each stock
REBALANCE = 5           # weekly
BULL_THRESH = 0.55      # >55% stocks above 200d SMA = bull
BEAR_THRESH = 0.40      # <40% = bear
EXPOSURE = 0.97
MIN_STOCKS = 50         # need at least this many stocks to compute breadth


class SP500BreadthTiming(Strategy):
    """Breadth-based QQQ/SPY/TLT-SHY rotation using % of SP500 above 200d SMA."""

    def __init__(
        self,
        trend_window: int = TREND_WINDOW,
        rebalance: int = REBALANCE,
        bull_thresh: float = BULL_THRESH,
        bear_thresh: float = BEAR_THRESH,
        exposure: float = EXPOSURE,
        min_stocks: int = MIN_STOCKS,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            rebalance=rebalance,
            bull_thresh=bull_thresh,
            bear_thresh=bear_thresh,
            exposure=exposure,
            min_stocks=min_stocks,
        )
        self.trend_window = int(trend_window)
        self.rebalance = int(rebalance)
        self.bull_thresh = float(bull_thresh)
        self.bear_thresh = float(bear_thresh)
        self.exposure = float(exposure)
        self.min_stocks = int(min_stocks)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute breadth: % of stocks above 200d SMA
        prices = ctx.closes_window(self.trend_window + 5)
        if len(prices) < self.trend_window:
            return []

        above_count = 0
        total_count = 0
        for sym in prices.columns:
            col = prices[sym].dropna()
            if len(col) < self.trend_window:
                continue
            sma = float(col.iloc[-self.trend_window:].mean())
            current = float(col.iloc[-1])
            if np.isfinite(sma) and np.isfinite(current) and sma > 0:
                total_count += 1
                if current > sma:
                    above_count += 1

        if total_count < self.min_stocks:
            # Can't compute breadth reliably
            target = {"SPY": self.exposure}
        else:
            breadth = above_count / total_count

            if breadth > self.bull_thresh:
                # Bull: hold QQQ
                target = {"QQQ": self.exposure}
            elif breadth < self.bear_thresh:
                # Bear: defensive TLT + SHY
                target = {}
                if "TLT" in closes_now.index:
                    target["TLT"] = 0.50
                if "SHY" in closes_now.index:
                    target["SHY"] = 0.47
                if not target and "SHY" in closes_now.index:
                    target["SHY"] = self.exposure
            else:
                # Neutral: hold SPY
                target = {"SPY": self.exposure}

        # Filter to available symbols
        target = {s: w for s, w in target.items() if s in closes_now.index}

        if not target:
            if "SHY" in closes_now.index:
                target = {"SHY": self.exposure}
            else:
                return []

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["QQQ", "SPY", "TLT", "SHY"]


NAME = "sp500_breadth_timing"
HYPOTHESIS = (
    "SP500 breadth-thrust timing: hold QQQ 97% when >55% of SP500 stocks above 200d SMA "
    "(bull breadth); TLT 50%+SHY 47% when <40% (bear); SPY 97% in neutral zone; "
    "weekly rebalance; breadth signal orthogonal to VIX, credit, and price momentum."
)

UNIVERSE = _universe

STRATEGY = SP500BreadthTiming()
