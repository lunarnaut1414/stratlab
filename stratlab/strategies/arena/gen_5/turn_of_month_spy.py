"""Turn-of-month equity tilt strategy (v4).

Hypothesis:
  Tilt SPY allocation upward during the last 3 and first 3 trading days of
  each month (the "TOM window"). Outside the window, hold SPY at a reduced
  allocation. The strategy stays in equities at all times — the TOM effect
  is captured as a tilt, not a binary switch.

Rationale: Studies by Ariel (1987) and Lakonishok & Smidt (1988) show most
stock returns occur in the TOM window. By holding SPY throughout but tilting
higher during the window, we avoid the drag of rotating into bonds while still
capturing the documented seasonal premium.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

DAYS_END = 3    # last N trading days of month = high-exposure window
DAYS_START = 3  # first N trading days of month = high-exposure window
HIGH_EXPOSURE = 0.95  # during TOM window (uptrend)
LOW_EXPOSURE = 0.65   # outside TOM window (uptrend)
DOWNTREND_HIGH = 0.70  # during TOM window (downtrend)
DOWNTREND_LOW = 0.35   # outside TOM window (downtrend)
TREND_WINDOW = 200     # bars for trend determination


class TurnOfMonthSpy(Strategy):
    """Tilt SPY up during turn-of-month; remain invested throughout."""

    def __init__(
        self,
        days_end: int = DAYS_END,
        days_start: int = DAYS_START,
        high_exposure: float = HIGH_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
        downtrend_high: float = DOWNTREND_HIGH,
        downtrend_low: float = DOWNTREND_LOW,
        trend_window: int = TREND_WINDOW,
    ) -> None:
        super().__init__(
            days_end=days_end,
            days_start=days_start,
            high_exposure=high_exposure,
            low_exposure=low_exposure,
            downtrend_high=downtrend_high,
            downtrend_low=downtrend_low,
            trend_window=trend_window,
        )
        self.days_end = int(days_end)
        self.days_start = int(days_start)
        self.high_exposure = float(high_exposure)
        self.low_exposure = float(low_exposure)
        self.downtrend_high = float(downtrend_high)
        self.downtrend_low = float(downtrend_low)
        self.trend_window = int(trend_window)

    def _is_turn_of_month(self, ts: pd.Timestamp, all_prior_dates: pd.DatetimeIndex) -> bool:
        """Determine if today is in the turn-of-month window."""
        cur_m = ts.month
        cur_y = ts.year

        cur_month_prior = all_prior_dates[
            (all_prior_dates.month == cur_m) & (all_prior_dates.year == cur_y)
        ]
        today_ordinal = len(cur_month_prior) + 1

        if today_ordinal <= self.days_start:
            return True

        if cur_m == 1:
            prev_m, prev_y = 12, cur_y - 1
        else:
            prev_m, prev_y = cur_m - 1, cur_y

        prev_month_dates = all_prior_dates[
            (all_prior_dates.month == prev_m) & (all_prior_dates.year == prev_y)
        ]
        if len(prev_month_dates) >= 15:
            est_total = len(prev_month_dates)
            if today_ordinal >= (est_total - self.days_end + 1):
                return True

        return False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < 35:
            return []

        spy_hist = ctx.history("SPY")
        if spy_hist is None or len(spy_hist) < 35:
            return []

        all_prior_dates = spy_hist.index
        in_window = self._is_turn_of_month(ctx.timestamp, all_prior_dates)

        # Trend filter: SPY above 200d SMA = uptrend
        spy_close_series = spy_hist["close"].dropna()
        in_uptrend = True
        if len(spy_close_series) >= self.trend_window:
            sma200 = float(spy_close_series.iloc[-self.trend_window:].mean())
            spy_last_close = float(spy_close_series.iloc[-1])
            in_uptrend = spy_last_close > sma200

        closes = ctx.closes()
        if closes.empty:
            return []

        spy_price = closes.get("SPY")
        if spy_price is None or not np.isfinite(float(spy_price)) or float(spy_price) <= 0:
            return []
        spy_price = float(spy_price)

        pv_dict = {"SPY": spy_price}
        portfolio_value = ctx.portfolio_value(pv_dict)
        if portfolio_value <= 0:
            return []

        if in_uptrend:
            target_exposure = self.high_exposure if in_window else self.low_exposure
        else:
            target_exposure = self.downtrend_high if in_window else self.downtrend_low
        target_shares = int(portfolio_value * target_exposure / spy_price)
        current_shares = int(ctx.position("SPY").size)
        delta = target_shares - current_shares

        # Only trade if delta is meaningful (>1% of current shares to avoid excessive churn)
        if abs(delta) < 1:
            return []

        orders: list[Order] = []
        if delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol="SPY"))
        else:
            orders.append(Order(side=OrderSide.BUY, size=abs(delta), symbol="SPY"))

        return orders


UNIVERSE = ["SPY"]

NAME = "turn_of_month_spy"
HYPOTHESIS = (
    "Turn-of-month equity tilt: increase SPY exposure to 95% during last 3 + first 3 trading "
    "days of each month (TOM window), reduce to 60% otherwise; always-invested approach to "
    "capture TOM seasonal premium without bond drag."
)

STRATEGY = TurnOfMonthSpy()
