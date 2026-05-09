"""Donchian channel breakout — classic trend-following template.

The original "Turtle" strategy boiled down to its essentials:

- Enter long when today's close breaks above the highest high of the
  prior ``entry_window`` bars.
- Exit when today's close breaks below the lowest low of the prior
  ``exit_window`` bars (typically shorter than entry_window — let
  winners run, cut losers fast).

Single-asset. Works on anything trending — futures, indices, individual
names with strong directional moves. Needs ``entry_window`` bars of
warmup before any signals fire.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class DonchianBreakout(Strategy):
    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        size: float = 100.0,
    ) -> None:
        super().__init__(
            entry_window=entry_window, exit_window=exit_window, size=size,
        )
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.size = size
        self.in_position = False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.entry_window:
            return []

        bars = ctx.history()
        close = bars["close"].iloc[-1]
        # Use prior bars (exclude today) so the breakout level isn't dragged by today's move.
        upper = bars["high"].iloc[-self.entry_window - 1 : -1].max()
        lower = bars["low"].iloc[-self.exit_window - 1 : -1].min()

        if not self.in_position and close > upper:
            self.in_position = True
            return [Order(side=OrderSide.BUY, size=self.size)]

        if self.in_position and close < lower:
            self.in_position = False
            return [Order(side=OrderSide.SELL, size=self.size)]

        return []

    def on_start(self) -> None:
        self.in_position = False
