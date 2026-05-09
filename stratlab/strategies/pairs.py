"""Pairs / mean-reversion on a hand-picked symbol pair.

Track the rolling z-score of the price ratio ``a / b``. When it
diverges past ``entry_z`` standard deviations from its mean, take
opposite positions (short the rich leg, long the cheap leg). Close
both legs when |z| reverts below ``exit_z``.

This is the simplest stat-arb pattern. It deliberately skips the
formal cointegration test (Engle-Granger / Johansen) — that requires
``statsmodels`` and gets fiddly. For a production system, pre-screen
your pairs offline using ADF on the residuals and only feed cointegrated
pairs into this strategy.

Use it on tight pairs: same sector, similar size, similar geography
(KO/PEP, GS/MS, V/MA, XOM/CVX, etc).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class Pairs(Strategy):
    def __init__(
        self,
        sym_a: str,
        sym_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        size: float = 100.0,
    ) -> None:
        super().__init__(
            sym_a=sym_a, sym_b=sym_b, lookback=lookback,
            entry_z=entry_z, exit_z=exit_z, size=size,
        )
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.size = size
        self.position_state = 0  # +1 long-A/short-B, -1 short-A/long-B, 0 flat

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.lookback:
            return []

        prices = ctx.closes_window(self.lookback)
        if self.sym_a not in prices.columns or self.sym_b not in prices.columns:
            return []

        ratio = (prices[self.sym_a] / prices[self.sym_b]).dropna()
        if len(ratio) < self.lookback or ratio.std() == 0:
            return []

        z = (ratio.iloc[-1] - ratio.mean()) / ratio.std()

        # Exit: ratio reverted toward mean
        if self.position_state != 0 and abs(z) < self.exit_z:
            orders = self._close_pair()
            self.position_state = 0
            return orders

        # Entry: ratio diverged
        if self.position_state == 0:
            if z > self.entry_z:
                # A rich, B cheap → short A / long B
                self.position_state = -1
                return [
                    Order(side=OrderSide.SELL, size=self.size, symbol=self.sym_a),
                    Order(side=OrderSide.BUY, size=self.size, symbol=self.sym_b),
                ]
            if z < -self.entry_z:
                # A cheap, B rich → long A / short B
                self.position_state = 1
                return [
                    Order(side=OrderSide.BUY, size=self.size, symbol=self.sym_a),
                    Order(side=OrderSide.SELL, size=self.size, symbol=self.sym_b),
                ]

        return []

    def _close_pair(self) -> list[Order]:
        if self.position_state == 1:
            return [
                Order(side=OrderSide.SELL, size=self.size, symbol=self.sym_a),
                Order(side=OrderSide.BUY, size=self.size, symbol=self.sym_b),
            ]
        if self.position_state == -1:
            return [
                Order(side=OrderSide.BUY, size=self.size, symbol=self.sym_a),
                Order(side=OrderSide.SELL, size=self.size, symbol=self.sym_b),
            ]
        return []

    def on_start(self) -> None:
        self.position_state = 0
