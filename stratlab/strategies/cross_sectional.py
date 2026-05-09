"""Cross-sectional long/short factor portfolio.

Generalizes the standard quant pattern: every ``rebalance`` bars, score
all currently-tradeable symbols by a user-supplied ``factor_fn``, then
go equal-weight long the top ``k`` and equal-weight short the bottom
``k``. Dollar-neutral by default.

The default ``factor_fn`` is 12-1 month momentum (Jegadeesh-Titman):
return over the last ~11 months, skipping the most recent month to
dodge short-term reversal. Swap it for any callable that takes a wide
``(lookback, n_symbols)`` close-price frame and returns a per-symbol
``Series`` — value, low-vol, residual-momentum, anything.

Example::

    def low_vol(prices):
        return -prices.pct_change().rolling(60).std().iloc[-1]

    strat = CrossSectionalFactor(factor_fn=low_vol, k=20, lookback=120)
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


def _momentum_12_1(prices: pd.DataFrame, skip: int = 21) -> pd.Series:
    """12-1 month total return (Jegadeesh-Titman)."""
    return prices.iloc[-skip] / prices.iloc[0] - 1.0


class CrossSectionalFactor(Strategy):
    def __init__(
        self,
        factor_fn: Callable[[pd.DataFrame], pd.Series] = _momentum_12_1,
        k: int = 20,
        lookback: int = 252,
        rebalance: int = 21,
        gross_leverage: float = 1.0,
        long_only: bool = False,
    ) -> None:
        super().__init__(
            k=k, lookback=lookback, rebalance=rebalance,
            gross=gross_leverage, long_only=long_only,
        )
        self.factor_fn = factor_fn
        self.k = k
        self.lookback = lookback
        self.rebalance = rebalance
        self.gross_leverage = gross_leverage
        self.long_only = long_only

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.lookback:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        prices = ctx.closes_window(self.lookback)
        scores = self.factor_fn(prices).dropna()
        min_names = self.k if self.long_only else 2 * self.k
        if len(scores) < min_names:
            return []

        ranked = scores.sort_values()
        longs = set(ranked.tail(self.k).index)
        shorts = set() if self.long_only else set(ranked.head(self.k).index)

        live_closes = ctx.closes()
        equity = ctx.portfolio_value({s: float(p) for s, p in live_closes.items()})
        n_legs = 1 if self.long_only else 2
        per_leg = equity * self.gross_leverage / n_legs
        per_name = per_leg / self.k

        target: dict[str, int] = {}
        for sym in longs:
            if sym in live_closes:
                target[sym] = int(per_name // float(live_closes[sym]))
        for sym in shorts:
            if sym in live_closes:
                target[sym] = -int(per_name // float(live_closes[sym]))

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, tgt in target.items():
            current = ctx.position(sym).size
            delta = tgt - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders
