"""SP500 breadth-gated sector ETF momentum rotation.

Hypothesis: hold top-3 SPDR sector ETFs by 42d momentum only when >55% of
SP500 stocks are above their 50d SMA (broad bull breadth); hold TLT when
breadth is 35-55% (moderate); SHY when breadth below 35% (bear breadth);
biweekly rebalance.

Rationale:
  - Market breadth (fraction of stocks above 50d SMA) measures broad
    participation in a bull market vs narrow leadership. When breadth is
    high, sector momentum strategies work better because the rising tide
    lifts all boats, and mean-reversion headwinds are minimal.
  - 50d SMA (vs 200d as in gen6_sp500_breadth_timing) gives a faster
    breadth signal that captures regime transitions earlier.
  - Routing to sector ETFs (vs QQQ) when breadth is high creates a
    different return profile that emphasizes the best-performing sectors
    rather than just tech/large-cap.
  - Degrading to TLT (not SPY) in moderate breadth avoids correlation
    with SPY-holding strategies.

Distinction from existing strategies:
  - gen6_sp500_breadth_timing uses 200d SMA breadth → QQQ/SPY/TLT.
    This uses 50d breadth → sector ETFs/TLT/SHY.
  - All sector ETF rotation strategies on leaderboard do NOT gate on
    breadth (they use VIX, credit, or SPY 200d SMA as gate).
  - This combines breadth + sector momentum in a novel way not seen
    in any gen_5 or gen_6 submission.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["XLK", "XLF", "XLY", "XLE", "XLI", "XLU", "XLV", "XLB", "XLP", "TLT", "SHY", "SPY"]


UNIVERSE = _universe

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 42       # ~2 months for sector momentum
BREADTH_WINDOW = 50        # 50d SMA for breadth
BULL_BREADTH = 0.55        # >55% above 50d SMA = bullish breadth
BEAR_BREADTH = 0.35        # <35% above 50d SMA = bearish breadth
EXPOSURE = 0.97
TOP_SECTORS = 3
MIN_STOCKS_FOR_BREADTH = 50  # need at least this many SP500 stocks with data

SECTOR_ETFS = ["XLK", "XLF", "XLY", "XLE", "XLI", "XLU", "XLV", "XLB", "XLP"]


class BreadthSectorMomentum(Strategy):
    """Breadth-gated sector ETF momentum: top-3 sectors when breadth strong, else bonds."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        bull_breadth: float = BULL_BREADTH,
        bear_breadth: float = BEAR_BREADTH,
        exposure: float = EXPOSURE,
        top_sectors: int = TOP_SECTORS,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            breadth_window=breadth_window,
            bull_breadth=bull_breadth,
            bear_breadth=bear_breadth,
            exposure=exposure,
            top_sectors=top_sectors,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.breadth_window = int(breadth_window)
        self.bull_breadth = float(bull_breadth)
        self.bear_breadth = float(bear_breadth)
        self.exposure = float(exposure)
        self.top_sectors = int(top_sectors)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.breadth_window, self.momentum_window) + 10
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

        # Compute breadth: fraction of SP500 stocks above 50d SMA
        prices = ctx.closes_window(self.breadth_window + 5)
        if len(prices) < self.breadth_window:
            return []

        above_count = 0
        total_count = 0
        for sym in prices.columns:
            # Only count actual stocks (not ETFs) for breadth
            # We identify sector ETFs and defensive ETFs to exclude from breadth calculation
            if sym in SECTOR_ETFS or sym in ["TLT", "SHY", "SPY", "QQQ", "IEF", "GLD", "AGG"]:
                continue
            col = prices[sym].dropna()
            if len(col) < self.breadth_window:
                continue
            sma = float(col.iloc[-self.breadth_window:].mean())
            current = float(col.iloc[-1])
            total_count += 1
            if current > sma:
                above_count += 1

        target: dict[str, float] = {}

        if total_count < MIN_STOCKS_FOR_BREADTH:
            # Not enough data — use SPY as fallback
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            breadth_frac = above_count / total_count

            if breadth_frac < self.bear_breadth:
                # Bear breadth: go to SHY (avoid TLT duration risk in early bear)
                if "SHY" in closes_now.index:
                    target["SHY"] = self.exposure

            elif breadth_frac < self.bull_breadth:
                # Moderate breadth: park in TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure

            else:
                # Bull breadth: hold top-K sectors by 42d momentum
                sector_prices = ctx.closes_window(self.momentum_window + 5)
                if len(sector_prices) < self.momentum_window:
                    if "TLT" in closes_now.index:
                        target["TLT"] = self.exposure
                else:
                    sector_scores: dict[str, float] = {}
                    for sym in SECTOR_ETFS:
                        if sym not in sector_prices.columns:
                            continue
                        col = sector_prices[sym].dropna()
                        if len(col) < self.momentum_window:
                            continue
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-self.momentum_window])
                        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                            continue
                        ret = p_end / p_start - 1.0
                        if np.isfinite(ret):
                            sector_scores[sym] = ret

                    if len(sector_scores) < 3:
                        if "TLT" in closes_now.index:
                            target["TLT"] = self.exposure
                    else:
                        k = min(self.top_sectors, len(sector_scores))
                        ranked = sorted(sector_scores, key=sector_scores.__getitem__, reverse=True)[:k]
                        per_weight = self.exposure / len(ranked)
                        for sym in ranked:
                            if sym in closes_now.index:
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


NAME = "breadth_sector_momentum"
HYPOTHESIS = (
    "SP500 breadth-gated sector ETF momentum rotation: hold top-3 SPDR sector ETFs by 42d "
    "momentum only when >55% of SP500 stocks are above their 50d SMA (broad bull breadth); "
    "hold TLT when breadth 35-55%; SHY when breadth below 35% (bear breadth); biweekly rebalance"
)

STRATEGY = BreadthSectorMomentum()
