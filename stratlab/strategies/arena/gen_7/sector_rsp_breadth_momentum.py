"""Sector ETF momentum with RSP equal-weight breadth gate.

Hypothesis: rank 11 SPDR sector ETFs by 63d return, hold top-3 equally
weighted only when RSP (equal-weight SP500 ETF) is above its 50d SMA
(broad market breadth signal); rotate all to IEF when breadth fails.
Rebalance every 10 bars.

Rationale: RSP vs SPY relative strength reveals whether the rally is broadly
participatory (RSP above trend = healthy breadth) or narrow/mega-cap-led.
When breadth is healthy, sector momentum is more reliable. When RSP is below
its 50d SMA, most stocks are under pressure — move to mid-duration bonds (IEF)
as a defensive shelter. This breadth gate is orthogonal to VIX-level and
credit-spread gates already on the leaderboard.

Distinction from existing strategies:
  - RSP 50d SMA breadth gate (not VIX level, not JNK credit signal)
  - Sector ETF rotation (not SP500 stocks, not single asset timing)
  - 63d momentum on sector ETFs (distinct from factor ETF rotation)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY", "XLRE", "XLC"
]
BREADTH_TICKER = "RSP"   # equal-weight S&P500 — breadth proxy
DEFENSIVE = "IEF"        # mid-duration treasury when breadth fails

REBALANCE_EVERY = 10     # bars (~2 weeks)
MOMENTUM_WINDOW = 63     # ~3 months
BREADTH_MA = 50          # 50d SMA on RSP
TOP_K = 3
EXPOSURE = 0.97
MIN_HOLD_BARS = 3


class SectorRspBreadthMomentum(Strategy):
    """Top-3 SPDR sector ETFs by 63d momentum when RSP above 50d SMA; else IEF."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        breadth_ma: int = BREADTH_MA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        min_hold_bars: int = MIN_HOLD_BARS,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            breadth_ma=breadth_ma,
            top_k=top_k,
            exposure=exposure,
            min_hold_bars=min_hold_bars,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.breadth_ma = int(breadth_ma)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.min_hold_bars = int(min_hold_bars)
        self._last_rebal = -999

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.breadth_ma) + 10
        if ctx.idx < warmup:
            return []

        # Enforce min hold period AND rebalance cadence
        bars_since_rebal = ctx.idx - self._last_rebal
        if bars_since_rebal < self.min_hold_bars:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # RSP breadth check
        try:
            rsp_hist = ctx.history(BREADTH_TICKER)
        except Exception:
            return []
        if rsp_hist is None or len(rsp_hist) < self.breadth_ma + 5:
            return []
        rsp_close = rsp_hist["close"].dropna()
        if len(rsp_close) < self.breadth_ma:
            return []
        rsp_sma = float(rsp_close.iloc[-self.breadth_ma:].mean())
        rsp_current = float(rsp_close.iloc[-1])
        breadth_ok = rsp_current > rsp_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not breadth_ok:
            # Defensive: IEF
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        else:
            # Rank sector ETFs by 63d momentum
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
                if DEFENSIVE in closes_now.index:
                    target[DEFENSIVE] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        self._last_rebal = ctx.idx

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


NAME = "sector_rsp_breadth_momentum"
HYPOTHESIS = (
    "Sector ETF momentum with RSP equal-weight breadth gate: rank 11 SPDR sector ETFs by 63d "
    "return, hold top-3 equally weighted only when RSP is above its 50d SMA; rotate to IEF "
    "when breadth fails; bi-weekly rebalance"
)

UNIVERSE = ["XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY",
            "XLRE", "XLC", "RSP", "IEF", "SPY"]

STRATEGY = SectorRspBreadthMomentum()
