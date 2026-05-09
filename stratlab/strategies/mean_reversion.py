from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bollinger Band mean reversion strategy (single-asset).

    Buys when price drops below lower band, sells when it rises above upper band.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, size: float = 100.0) -> None:
        super().__init__(window=window, num_std=num_std, size=size)
        self.window = window
        self.num_std = num_std
        self.size = size
        self.in_position = False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.window:
            return []

        closes = ctx.history()["close"].iloc[-self.window :]
        mean = float(closes.mean())
        std = float(closes.std())

        if std == 0:
            return []

        upper = mean + self.num_std * std
        lower = mean - self.num_std * std
        price = float(ctx.bar()["close"])

        if price < lower and not self.in_position:
            self.in_position = True
            return [Order(side=OrderSide.BUY, size=self.size)]

        if price > upper and self.in_position:
            self.in_position = False
            return [Order(side=OrderSide.SELL, size=self.size)]

        return []

    def on_start(self) -> None:
        self.in_position = False
