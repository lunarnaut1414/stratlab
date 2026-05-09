from __future__ import annotations

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.strategies.base import Strategy


class SMACrossover(Strategy):
    """Simple moving average crossover strategy.

    Buys when fast SMA crosses above slow SMA, sells when it crosses below.
    """

    def __init__(self, fast: int = 10, slow: int = 30, size: float = 100.0) -> None:
        super().__init__(fast=fast, slow=slow, size=size)
        self.fast = fast
        self.slow = slow
        self.size = size
        self.in_position = False

    def on_bar(self, idx: int, history: pd.DataFrame) -> list[Order]:
        if idx < self.slow:
            return []

        closes = history["close"].iloc[: idx + 1]
        fast_sma = closes.iloc[-self.fast :].mean()
        slow_sma = closes.iloc[-self.slow :].mean()

        prev_closes = closes.iloc[:-1]
        if len(prev_closes) < self.slow:
            return []
        prev_fast = prev_closes.iloc[-self.fast :].mean()
        prev_slow = prev_closes.iloc[-self.slow :].mean()

        # crossover: fast crosses above slow
        if prev_fast <= prev_slow and fast_sma > slow_sma and not self.in_position:
            self.in_position = True
            return [Order(side=OrderSide.BUY, size=self.size)]

        # crossunder: fast crosses below slow
        if prev_fast >= prev_slow and fast_sma < slow_sma and self.in_position:
            self.in_position = False
            return [Order(side=OrderSide.SELL, size=self.size)]

        return []

    def on_start(self) -> None:
        self.in_position = False
