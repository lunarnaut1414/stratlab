from __future__ import annotations

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class Momentum(Strategy):
    """RSI-based momentum strategy (single-asset).

    Buys when RSI drops below oversold threshold, sells when it rises above overbought.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        size: float = 100.0,
    ) -> None:
        super().__init__(period=period, oversold=oversold, overbought=overbought, size=size)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.size = size
        self.in_position = False

    def _rsi(self, closes: pd.Series) -> float:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(self.period).mean().iloc[-1]
        avg_loss = loss.rolling(self.period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.period + 1:
            return []

        closes = ctx.history()["close"]
        rsi = self._rsi(closes)

        if rsi < self.oversold and not self.in_position:
            self.in_position = True
            return [Order(side=OrderSide.BUY, size=self.size)]

        if rsi > self.overbought and self.in_position:
            self.in_position = False
            return [Order(side=OrderSide.SELL, size=self.size)]

        return []

    def on_start(self) -> None:
        self.in_position = False
