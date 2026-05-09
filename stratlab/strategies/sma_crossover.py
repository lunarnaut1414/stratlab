from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class SMACrossover(Strategy):
    """Simple moving average crossover strategy (single-asset).

    Buys when fast SMA crosses above slow SMA, sells when it crosses below.
    """

    def __init__(self, fast: int = 10, slow: int = 30, size: float = 100.0) -> None:
        super().__init__(fast=fast, slow=slow, size=size)
        self.fast = fast
        self.slow = slow
        self.size = size
        self.in_position = False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # history() returns bars before today, so we need slow+1 of them
        # to compute both the latest SMA and the prior-bar SMA for crossover.
        if ctx.idx < self.slow + 1:
            return []

        closes = ctx.history()["close"]
        fast_sma = closes.iloc[-self.fast :].mean()
        slow_sma = closes.iloc[-self.slow :].mean()

        prev_closes = closes.iloc[:-1]
        if len(prev_closes) < self.slow:
            return []
        prev_fast = prev_closes.iloc[-self.fast :].mean()
        prev_slow = prev_closes.iloc[-self.slow :].mean()

        if prev_fast <= prev_slow and fast_sma > slow_sma and not self.in_position:
            self.in_position = True
            return [Order(side=OrderSide.BUY, size=self.size)]

        if prev_fast >= prev_slow and fast_sma < slow_sma and self.in_position:
            self.in_position = False
            return [Order(side=OrderSide.SELL, size=self.size)]

        return []

    def on_start(self) -> None:
        self.in_position = False
