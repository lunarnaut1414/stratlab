"""SP500 with Dynamic Defensive Rotation — gen_7 sonnet-7

Hypothesis: When SPY is above 200d SMA (bull), hold top-15 SP500 stocks
by 63d momentum, equal-weight. When SPY is below 200d SMA (bear), hold
whichever of TLT/IEF/GLD has highest 42d momentum (single asset, 97%).

Rationale: Most strategies use static defensive allocations (always TLT, or
TLT+SHY 50/50). During bear markets, the best-performing defensive asset
varies: sometimes TLT wins (deflation), sometimes GLD wins (stagflation),
sometimes IEF is best (moderate recession). By dynamically picking the best
defensive asset based on recent momentum, this strategy should capture more
defensive alpha than static TLT allocation.

Monthly rebalance (21 bars) for the offensive leg; defensive can rotate
every 10 bars to capture faster safe-haven shifts.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_OFFENSIVE = 21     # monthly for stock selection
REBALANCE_DEFENSIVE = 10     # biweekly for defensive rotation
MOMENTUM_WINDOW = 63         # 3 months for offensive
DEF_MOMENTUM_WINDOW = 42     # 6 weeks for defensive rotation
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_DEFENSIVE = ["TLT", "IEF", "GLD"]


class SP500DynDefensiveRotation(Strategy):
    """SP500 momentum in bull; dynamic TLT/IEF/GLD best-momentum in bear."""

    def __init__(
        self,
        rebalance_offensive: int = REBALANCE_OFFENSIVE,
        rebalance_defensive: int = REBALANCE_DEFENSIVE,
        momentum_window: int = MOMENTUM_WINDOW,
        def_momentum_window: int = DEF_MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_offensive=rebalance_offensive,
            rebalance_defensive=rebalance_defensive,
            momentum_window=momentum_window,
            def_momentum_window=def_momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_offensive = int(rebalance_offensive)
        self.rebalance_defensive = int(rebalance_defensive)
        self.momentum_window = int(momentum_window)
        self.def_momentum_window = int(def_momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 10
        if ctx.idx < warmup:
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
            # Defensive: pick best of TLT/IEF/GLD by 42d momentum
            # Rebalance defensive every REBALANCE_DEFENSIVE bars
            if ctx.idx % self.rebalance_defensive != 0:
                return []

            prices = ctx.closes_window(self.def_momentum_window + 5)
            if len(prices) < self.def_momentum_window:
                return []

            def_scores: dict[str, float] = {}
            for sym in _DEFENSIVE:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.def_momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.def_momentum_window])
                if p_start <= 0:
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    def_scores[sym] = ret

            if def_scores:
                best_def = max(def_scores, key=def_scores.__getitem__)
                if best_def in live:
                    target[best_def] = self.exposure
            else:
                # Fallback: TLT
                if "TLT" in live:
                    target["TLT"] = self.exposure
        else:
            # Bull: top-K SP500 momentum stocks; rebalance monthly
            if ctx.idx % self.rebalance_offensive != 0:
                return []

            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in _DEFENSIVE or sym == _SPY:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0:
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 5:
                if "TLT" in live:
                    target["TLT"] = self.exposure
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
    return sp500_tickers() + ["TLT", "IEF", "GLD", "SPY"]


NAME = "sp500_dyn_defensive_rotation"
HYPOTHESIS = (
    "Defensive rotation within bear markets: when SPY below 200d SMA hold whichever of "
    "TLT/IEF/GLD has highest 42d momentum (1 asset only, 97%); when SPY above 200d SMA "
    "hold top-15 SP500 by 63d momentum equal-weight; monthly rebalance; dynamic bond/gold "
    "rotation as the defensive bucket instead of static TLT"
)

UNIVERSE = _universe

STRATEGY = SP500DynDefensiveRotation()
