"""Trend strategy gated by news sentiment.

The "hello world" demonstrating how :func:`stratlab.daily_sentiment`
features wire into a backtest. Only goes long when **both**:

1. Price momentum is positive (close > N-day SMA), and
2. The day's net sentiment for a chosen topic is above ``sentiment_threshold``.

Goes flat when either condition fails. Optionally goes short when both
are negative (``allow_short=True``).

The sentiment series is supplied at construction so the same overlay
mechanic works with any signal — global business sentiment, a custom
weighted blend across sources, or even a non-news indicator. The series
just needs to be date-indexed; missing dates are treated as ``0`` (no
signal, neither bullish nor bearish).

Example::

    from stratlab import daily_sentiment, Backtest, load_bars
    from stratlab.strategies.news_overlay import NewsOverlay

    sent = daily_sentiment(start="2020-01-01", topics=["business"])
    # collapse to one number per day
    daily = sent.mean(axis=1)

    data = load_bars("SPY", start="2020-01-01")
    strat = NewsOverlay(sentiment=daily, momentum_window=20)
    result = Backtest(data={"SPY": data}, strategy=strat).run()
"""
from __future__ import annotations

import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


class NewsOverlay(Strategy):
    def __init__(
        self,
        sentiment: pd.Series,
        momentum_window: int = 20,
        sentiment_threshold: float = 0.0,
        size: float = 100.0,
        allow_short: bool = False,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            sentiment_threshold=sentiment_threshold,
            size=size, allow_short=allow_short,
        )
        # Normalize index to date-only timestamps for cheap exact lookup.
        s = sentiment.copy()
        s.index = pd.to_datetime(s.index).normalize()
        self.sentiment = s.sort_index()
        self.momentum_window = momentum_window
        self.sentiment_threshold = sentiment_threshold
        self.size = size
        self.allow_short = allow_short
        self.position_state = 0  # +1 long, -1 short, 0 flat

    def _sentiment_as_of(self, ts: pd.Timestamp) -> float:
        """Most recent sentiment value strictly before ``ts`` — i.e., the
        last published-day's score we could have observed before today's
        bar opens. Avoids leaking today's news into today's decision."""
        cutoff = ts.normalize()
        prior = self.sentiment.loc[self.sentiment.index < cutoff]
        if prior.empty:
            return 0.0
        v = prior.iloc[-1]
        try:
            return float(v) if pd.notna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.momentum_window:
            return []

        closes = ctx.history()["close"]
        sma = closes.iloc[-self.momentum_window:].mean()
        price = closes.iloc[-1]
        momentum_up = price > sma
        sent = self._sentiment_as_of(ctx.timestamp)

        bullish = momentum_up and sent > self.sentiment_threshold
        bearish = (not momentum_up) and sent < -self.sentiment_threshold

        target = 0
        if bullish:
            target = 1
        elif bearish and self.allow_short:
            target = -1

        if target == self.position_state:
            return []

        orders: list[Order] = []
        # Close current leg if any
        if self.position_state == 1:
            orders.append(Order(side=OrderSide.SELL, size=self.size))
        elif self.position_state == -1:
            orders.append(Order(side=OrderSide.BUY, size=self.size))
        # Open new leg
        if target == 1:
            orders.append(Order(side=OrderSide.BUY, size=self.size))
        elif target == -1:
            orders.append(Order(side=OrderSide.SELL, size=self.size))
        self.position_state = target
        return orders

    def on_start(self) -> None:
        self.position_state = 0
