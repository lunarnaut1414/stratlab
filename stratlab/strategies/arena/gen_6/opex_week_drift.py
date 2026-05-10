"""Options-expiration week drift strategy — gen_6 sonnet-2

Hypothesis:
  Buy QQQ at 97% exposure on the Monday (or first trading day) of the
  monthly options expiration week (the week containing the 3rd Friday of
  each month). Hold through Thursday (4 trading days). Hold SHY otherwise.

  With a SPY 200d SMA gate: reduce to 60% QQQ in downtrend (SPY below 200d)
  to limit exposure in bear markets.

Rationale:
  Options expiration week exhibits a positive drift in equity indices,
  documented in academic literature (e.g. "The Pre-FOMC Announcement Drift"
  analog for options). Market makers unwind hedges as gamma exposure declines
  throughout expiration week, creating systematic buying pressure. This is
  calendar-driven and has near-zero correlation to momentum, credit-spread,
  trend-following, or VIX-based timing strategies.

  Key design choices:
  - QQQ (higher beta than SPY) to maximize the drift effect
  - 4-day hold = buy Mon, sell Fri morning = capture Mon-Thu drift
  - 200d SMA gate reduces exposure in structural downtrends
  - SHY as default (no interest rate risk when not in trade)

Structural distinctions vs existing leaderboard:
  - Pure calendar signal (no momentum, no credit, no VIX)
  - Very low turnover (12 entry/exits per year)
  - Different daily return pattern → low correlation expected
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

TREND_WINDOW = 200
BULL_EXPOSURE = 0.97
BEAR_EXPOSURE = 0.60
RISK_ASSET = "QQQ"
DEFENSIVE = "SHY"


def _is_opex_week(ts: pd.Timestamp) -> bool:
    """Return True if ts falls in the week containing the 3rd Friday of its month."""
    # Find the 3rd Friday of the month
    year, month = ts.year, ts.month
    # Start from the 1st of the month
    first = pd.Timestamp(year=year, month=month, day=1)
    # Find first Friday (weekday 4)
    days_to_friday = (4 - first.weekday()) % 7
    first_friday = first + pd.Timedelta(days=days_to_friday)
    third_friday = first_friday + pd.Timedelta(weeks=2)
    # OpEx week = Monday through Friday of the week containing 3rd Friday
    # Monday of that week:
    opex_monday = third_friday - pd.Timedelta(days=third_friday.weekday())
    opex_friday = opex_monday + pd.Timedelta(days=4)
    return opex_monday <= ts <= opex_friday


class OpExWeekDrift(Strategy):
    """Buy QQQ during monthly OpEx week; SHY otherwise."""

    def __init__(
        self,
        trend_window: int = TREND_WINDOW,
        bull_exposure: float = BULL_EXPOSURE,
        bear_exposure: float = BEAR_EXPOSURE,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            bull_exposure=bull_exposure,
            bear_exposure=bear_exposure,
        )
        self.trend_window = int(trend_window)
        self.bull_exposure = float(bull_exposure)
        self.bear_exposure = float(bear_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.trend_window + 5:
            return []

        # Get SPY trend
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            spy_hist = None

        bull = True
        if spy_hist is not None and len(spy_hist) >= self.trend_window + 5:
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                sma = float(spy_close.iloc[-self.trend_window:].mean())
                bull = float(spy_close.iloc[-1]) > sma

        # Calendar check: is today in OpEx week?
        in_opex = _is_opex_week(ctx.timestamp)

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target
        target: dict[str, float] = {}
        if in_opex:
            exp = self.bull_exposure if bull else self.bear_exposure
            if RISK_ASSET in closes_now.index:
                target[RISK_ASSET] = exp
            else:
                if DEFENSIVE in closes_now.index:
                    target[DEFENSIVE] = self.bull_exposure
        else:
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.bull_exposure

        if not target:
            return []

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "opex_week_drift"
HYPOTHESIS = (
    "Options-expiration week drift: buy QQQ at 97% on Monday of monthly OpEx week "
    "(week containing 3rd Friday); hold through Thursday; SHY otherwise; "
    "60% QQQ if SPY below 200d SMA; captures OpEx week positive drift with near-zero "
    "correlation to momentum and credit-spread timing signals."
)

UNIVERSE = [RISK_ASSET, DEFENSIVE, "SPY"]

STRATEGY = OpExWeekDrift()
