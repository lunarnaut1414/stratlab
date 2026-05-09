"""Long top-K momentum names with an inverse-ETF hedge.

Combines two ideas:

1. **Long-only 12-1 momentum** on a stock universe. Equal-weight long
   the top ``k`` names by Jegadeesh-Titman 12-1 return.
2. **Tactical inverse-ETF hedge.** When a market-trend signal flips
   bearish (default: SPY < 200-day SMA), allocate ``hedge_fraction`` of
   equity into a long position in an inverse ETF (default: SH, the
   −1× SPY ETF). When bullish, fully invested in long names; no hedge.

Avoids the "short individual names" path entirely — every position the
engine holds is *long*. The "short the index" exposure comes from a
long position in an inverse-correlated ETF whose price already
incorporates the daily-compounding decay.

Required data:

- Each name in the long universe (e.g., ``sp500_tickers()``)
- The ``trend_symbol`` (default ``"SPY"``) — used purely for the
  trend signal, not traded
- The ``hedge_symbol`` (default ``"SH"``) — held when bearish
"""
from __future__ import annotations

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class MomentumPlusInverse(Strategy):
    def __init__(
        self,
        k: int = 20,
        lookback: int = 252,
        rebalance: int = 21,
        skip: int = 21,
        trend_symbol: str = "SPY",
        trend_window: int = 200,
        hedge_symbol: str = "SH",
        hedge_fraction: float = 0.25,
    ) -> None:
        super().__init__(
            k=k, lookback=lookback, rebalance=rebalance, skip=skip,
            trend_symbol=trend_symbol, trend_window=trend_window,
            hedge_symbol=hedge_symbol, hedge_fraction=hedge_fraction,
        )
        self.k = k
        self.lookback = lookback
        self.rebalance = rebalance
        self.skip = skip
        self.trend_symbol = trend_symbol
        self.trend_window = trend_window
        self.hedge_symbol = hedge_symbol
        self.hedge_fraction = hedge_fraction

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < max(self.lookback, self.trend_window) + 1:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # ---- 1. Trend signal: SPY vs its trend_window-day SMA ----
        trend_close = ctx.history(self.trend_symbol)["close"]
        if len(trend_close) < self.trend_window:
            return []
        trend_sma = trend_close.iloc[-self.trend_window:].mean()
        trend_price = trend_close.iloc[-1]
        bearish = trend_price < trend_sma

        # ---- 2. Long top-K by 12-1 momentum, excluding utility tickers ----
        prices = ctx.closes_window(self.lookback)
        prices = prices.drop(
            columns=[self.trend_symbol, self.hedge_symbol], errors="ignore",
        )
        scores = (prices.iloc[-self.skip] / prices.iloc[0] - 1.0).dropna()
        if len(scores) < self.k:
            return []
        ranked = scores.sort_values()
        longs = list(ranked.tail(self.k).index)

        # ---- 3. Sizing: split equity between longs and (optional) hedge ----
        live_closes = ctx.closes()
        equity = ctx.portfolio_value(
            {s: float(p) for s, p in live_closes.items()}
        )
        long_fraction = 1.0 - self.hedge_fraction if bearish else 1.0
        per_name = (equity * long_fraction) / self.k

        target: dict[str, int] = {}
        for sym in longs:
            if sym in live_closes:
                target[sym] = int(per_name // float(live_closes[sym]))

        if bearish and self.hedge_symbol in live_closes:
            hedge_capital = equity * self.hedge_fraction
            target[self.hedge_symbol] = int(
                hedge_capital // float(live_closes[self.hedge_symbol])
            )

        # ---- 4. Diff vs current positions, emit orders ----
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
