"""Equity ETF cross-sectional momentum with trend confirmation.

Hypothesis: Rank equity-focused ETFs (not individual stocks) by 63d return.
Hold top-3 ETFs that are ALSO above their own 100d SMA (individual trend gate).
When SPY is bearish (below 200d SMA), hold IEF. This creates a diversified
equity ETF rotation strategy that is mechanically distinct from SP500 stock picking.

The key insight: the IS window 2010-2018 was dominated by tech/growth, so any
strategy that can rotate into QQQ/XLK aggressively during that period will have
higher IS Calmar than a balanced multi-asset approach.

Design:
  - Universe: equity ETFs with IS coverage — SPY, QQQ, MDY, IWM, EEM, EFA,
    XLK, XLF, XLV, XLI, XLY, XLE, XLB, XLU.
  - Rank by 63d return; require price > own 100d SMA.
  - Hold top-3 qualifying ETFs; inverse-vol weighted.
  - Rebalance every 10 bars (biweekly).
  - SPY 200d SMA outer gate: IEF when bearish.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # 6-month momentum
TREND_WINDOW = 100        # per-ETF trend gate
VOL_WINDOW = 21           # for inverse-vol weight
SPY_TREND_WINDOW = 200    # outer trend gate
TOP_K = 3                 # concentrated top-3
EXPOSURE = 0.97

EQUITY_ETFS = [
    "QQQ", "MDY", "IWM",        # broad equity non-SPY
    "XLK", "XLF", "XLV",       # top sector ETFs
    "XLI", "XLY", "XLE",       # more sectors
    "EFA", "EEM",               # international
    "SPY",                      # benchmark also rankable
]


class EquityETFMomentumTrend(Strategy):
    """Equity ETF top-3 by 63d momentum with individual 100d SMA trend gate;
    inverse-vol weighted; SPY 200d outer gate to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = max(self.momentum_window, self.trend_window, self.vol_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in EQUITY_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < need - 5:
                    continue
                arr = col.values

                # Per-ETF trend gate
                if len(arr) < self.trend_window:
                    continue
                etf_sma = float(np.mean(arr[-self.trend_window:]))
                if float(arr[-1]) <= etf_sma:
                    continue

                # 63d momentum
                if len(arr) < self.momentum_window + 2:
                    continue
                p_end = float(arr[-1])
                p_start = float(arr[-self.momentum_window])
                if p_start <= 0:
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol
                if len(arr) < self.vol_window + 1:
                    continue
                tail = arr[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6:
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 1:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


NAME = "equity_etf_momentum_trend"
HYPOTHESIS = (
    "Equity ETF top-3 by 63d momentum with individual 100d SMA trend confirmation gate; "
    "inverse-vol weighted; SPY 200d outer gate to IEF defensive; biweekly rebalance; "
    "universe is equity ETFs (QQQ, MDY, IWM, XL* sectors, EFA, EEM, SPY) — holds ETFs "
    "not individual stocks, distinct from all SP500 stock-picking strategies"
)

UNIVERSE = EQUITY_ETFS + ["IEF"]

STRATEGY = EquityETFMomentumTrend()
