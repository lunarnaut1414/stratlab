"""opus-1 mutation of turn_of_month_spy (parent IS Calmar 0.51, corr 0.68).

Pre-FOMC drift seasonal — a different calendar pattern entirely:
  - Window:       last 3 + first 3 trading days of the month  ->  the 5
                  trading days before each scheduled FOMC meeting,
                  plus 1 trading day after the announcement.
  - FOMC dates:   FOMC meetings cluster on the 3rd Tuesday/Wednesday of each
                  Mar / Jun / Sep / Dec (quarterly meetings); the strategy
                  approximates the schedule from the calendar without
                  hard-coding event dates. Approximation is conservative — it
                  uses the 3rd Wed of Mar/Jun/Sep/Dec.

  - Universe / exposure: SPY tilt (high 95%, low 60%) — same pattern but
                          driven by the FOMC calendar instead of TOM.
  - Trend filter: SPY 200d SMA  ->  SPY 100d SMA (faster, since FOMC effect
                                     is more about meeting cadence than
                                     long-term trend).
  - Rebalance:    daily         ->  daily (kept).

The signal is structurally different: TOM fires ~12 windows/year (~6 days each
= 72 high-exposure days/year); FOMC drift fires only 4 windows/year (~6 days
each = 24 high-exposure days/year). Different days = different daily-return
path even on identical SPY exposure.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY"]

DAYS_BEFORE = 5
DAYS_AFTER = 1
HIGH_EXPOSURE = 0.95
LOW_EXPOSURE = 0.60
DOWNTREND_HIGH = 0.70
DOWNTREND_LOW = 0.30
TREND_WINDOW = 100


class FomcDriftSeasonal(Strategy):
    def __init__(
        self,
        days_before: int = DAYS_BEFORE,
        days_after: int = DAYS_AFTER,
        high_exposure: float = HIGH_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
        downtrend_high: float = DOWNTREND_HIGH,
        downtrend_low: float = DOWNTREND_LOW,
        trend_window: int = TREND_WINDOW,
    ) -> None:
        super().__init__(
            days_before=days_before,
            days_after=days_after,
            high_exposure=high_exposure,
            low_exposure=low_exposure,
            downtrend_high=downtrend_high,
            downtrend_low=downtrend_low,
            trend_window=trend_window,
        )
        self.days_before = int(days_before)
        self.days_after = int(days_after)
        self.high_exposure = float(high_exposure)
        self.low_exposure = float(low_exposure)
        self.downtrend_high = float(downtrend_high)
        self.downtrend_low = float(downtrend_low)
        self.trend_window = int(trend_window)

    @staticmethod
    def _third_wed(year: int, month: int) -> pd.Timestamp:
        """Return the date of the 3rd Wednesday of (year, month)."""
        cal = calendar.Calendar(firstweekday=0)
        weds = [d for d in cal.itermonthdates(year, month)
                if d.month == month and d.weekday() == 2]
        return pd.Timestamp(weds[2])

    def _is_in_fomc_window(self, ts: pd.Timestamp, all_dates: pd.DatetimeIndex) -> bool:
        """True if ts is in the [-days_before, +days_after] trading-day window
        around any quarterly FOMC date (3rd Wed of Mar/Jun/Sep/Dec) in the
        current or adjacent year."""
        candidates: list[pd.Timestamp] = []
        for yr in (ts.year - 1, ts.year, ts.year + 1):
            for m in (3, 6, 9, 12):
                try:
                    candidates.append(self._third_wed(yr, m))
                except IndexError:
                    pass

        # Convert each candidate to nearest trading-day index in all_dates;
        # then mark a window of [-days_before, +days_after] trading days.
        if len(all_dates) == 0:
            return False
        td_index = pd.DatetimeIndex(all_dates)
        ts_pos_arr = td_index.get_indexer([ts], method="nearest")
        ts_pos = int(ts_pos_arr[0]) if len(ts_pos_arr) > 0 else -1
        if ts_pos < 0:
            return False

        for c in candidates:
            pos_arr = td_index.get_indexer([c], method="nearest")
            if len(pos_arr) == 0:
                continue
            cpos = int(pos_arr[0])
            if cpos < 0:
                continue
            # Trading-day window around FOMC date.
            if cpos - self.days_before <= ts_pos <= cpos + self.days_after:
                return True
        return False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.trend_window + 5:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if spy_hist is None or len(spy_hist) < self.trend_window + 1:
            return []

        all_dates = spy_hist.index
        in_window = self._is_in_fomc_window(ctx.timestamp, all_dates)

        spy_close = spy_hist["close"].dropna()
        sma = float(spy_close.iloc[-self.trend_window:].mean())
        in_uptrend = float(spy_close.iloc[-1]) > sma

        live = ctx.closes()
        if live.empty:
            return []
        spy_price = live.get("SPY")
        if spy_price is None or not np.isfinite(float(spy_price)) or float(spy_price) <= 0:
            return []
        spy_price = float(spy_price)
        equity = ctx.portfolio_value({"SPY": spy_price})
        if equity <= 0:
            return []

        if in_uptrend:
            target_exposure = self.high_exposure if in_window else self.low_exposure
        else:
            target_exposure = self.downtrend_high if in_window else self.downtrend_low
        target_shares = int(equity * target_exposure / spy_price)
        cur_shares = int(ctx.position("SPY").size)
        delta = target_shares - cur_shares
        if abs(delta) < 1:
            return []

        orders: list[Order] = []
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(delta), symbol="SPY"))
        else:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol="SPY"))
        return orders


NAME = "opus1_fomc_drift_seasonal"
HYPOTHESIS = (
    "Mutate turn_of_month_spy: pre-FOMC drift seasonal — tilt SPY 95% during 5 "
    "trading days before quarterly FOMC dates (3rd Wed of Mar/Jun/Sep/Dec) "
    "and 1 day after; 60% otherwise; trend-aware reductions when SPY<100d."
)

STRATEGY = FomcDriftSeasonal()
