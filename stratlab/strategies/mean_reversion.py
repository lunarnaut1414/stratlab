from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class MeanReversion(Strategy):
    """Bollinger Band mean reversion (single-asset).

    Submits a BUY limit at the lower band when flat, and a SELL limit
    at the upper band when long. The limit either fills today (if
    today's range crosses the band) or is dropped — perfect use of
    limit orders since we're explicitly trying to enter at a price
    extreme. Showcases the limit-intraday execution model.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, size: float = 100.0) -> None:
        super().__init__(window=window, num_std=num_std, size=size)
        self.window = window
        self.num_std = num_std
        self.size = size

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

        symbol = next(iter(ctx._aligned))
        position_size = ctx.position(symbol).size

        if position_size == 0:
            return [Order(side=OrderSide.BUY, size=self.size, limit_price=lower)]
        if position_size > 0:
            return [Order(side=OrderSide.SELL, size=self.size, limit_price=upper)]
        return []
