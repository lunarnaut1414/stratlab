"""SPDR Sector ETF Momentum Rotation — gen_8 sonnet-10

Hypothesis: Rank the 9 available SPDR sector ETFs (XLK, XLV, XLF, XLE, XLI,
XLU, XLY, XLP, XLB) by their 42-day return. Hold top-3 equally weighted when
SPY is above its 200d SMA (bull regime). Rotate to IEF 97% when SPY below 200d
SMA (bear regime).

Rationale: Cross-sector relative momentum at the ETF level should be less
correlated to individual-stock-selection strategies that dominate the leaderboard.
By rotating among the 9 legacy SPDR sectors (all with long history), we capture
sector-rotation alpha without the idiosyncratic noise of individual stocks.
The 42-day window is shorter than typical 3-6 month stock momentum to reflect
faster sector-level regime shifts.

Biweekly rebalance (10 bars) to generate sufficient trade count.
IEF as defensive (7-10yr treasury) — different from TLT or SHY.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 42      # ~2 months
TREND_WINDOW = 200        # SPY 200d gate
TOP_K = 3                 # top-3 sectors
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"

# The 9 legacy SPDR sector ETFs — all have data back to 1998-2000+
_SECTORS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLU", "XLY", "XLP", "XLB"]


class SectorETFMomentumRotation(Strategy):
    """Top-3 SPDR sector ETFs by 42d momentum, SPY-trend gated."""

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
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
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
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Rank sectors by 42d momentum
            scores: dict[str, float] = {}
            for sector in _SECTORS:
                try:
                    hist = ctx.history(sector)
                except (KeyError, Exception):
                    continue
                if len(hist) < self.momentum_window + 2:
                    continue
                close_col = hist["close"].dropna()
                if len(close_col) < self.momentum_window:
                    continue
                ret = float(close_col.iloc[-1] / close_col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret) and sector in live:
                    scores[sector] = ret

            if len(scores) < 3:
                # Not enough sectors — hold SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
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


UNIVERSE = _SECTORS + [_IEF, _SPY]

NAME = "sector_etf_momentum_rotation"
HYPOTHESIS = (
    "SPDR sector ETF momentum rotation: rank all 9 available SPDR sector ETFs "
    "(XLK,XLV,XLF,XLE,XLI,XLU,XLY,XLP,XLB) by 42-day return, hold top-3 equally weighted "
    "when SPY is above 200d SMA; rotate to IEF 97% when SPY below 200d SMA; "
    "biweekly rebalance; pure sector-level momentum without individual stock selection"
)

STRATEGY = SectorETFMomentumRotation()
