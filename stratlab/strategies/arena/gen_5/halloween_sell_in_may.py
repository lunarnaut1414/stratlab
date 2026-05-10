"""Halloween / Sell-in-May seasonal strategy.

Hypothesis:
  Hold SPY from November 1 through April 30 (winter months) and rotate to TLT
  from May 1 through October 31 (summer months). Pure calendar seasonality
  exploiting the well-documented equity risk premium asymmetry across seasons.

Rationale: The "Sell in May and Go Away" effect (also called the Halloween
effect) has been documented in dozens of markets over 100+ years. Bouman &
Jacobsen (2002) demonstrated that equity returns are significantly higher
Nov-Apr than May-Oct in 36 of 37 countries studied. The premium persists even
after transaction costs. A simple implementation here routes the summer months
into TLT (intermediate Treasuries) rather than cash to capture fixed income
carry during the equity underperformance period.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# May through October inclusive = summer (hold TLT)
SUMMER_MONTHS = {5, 6, 7, 8, 9, 10}
# November through April inclusive = winter (hold SPY)
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}


class HalloweenSellInMay(Strategy):
    """Hold SPY Nov-Apr, TLT May-Oct."""

    def __init__(
        self,
        spy_exposure: float = 0.95,
        tlt_exposure: float = 0.95,
    ) -> None:
        super().__init__(
            spy_exposure=spy_exposure,
            tlt_exposure=tlt_exposure,
        )
        self.spy_exposure = spy_exposure
        self.tlt_exposure = tlt_exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < 5:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        spy_price = closes.get("SPY")
        tlt_price = closes.get("TLT")

        if spy_price is None or not np.isfinite(float(spy_price)) or float(spy_price) <= 0:
            return []
        if tlt_price is None or not np.isfinite(float(tlt_price)) or float(tlt_price) <= 0:
            return []

        spy_price = float(spy_price)
        tlt_price = float(tlt_price)

        live_prices = {"SPY": spy_price, "TLT": tlt_price}
        portfolio_value = ctx.portfolio_value(live_prices)
        if portfolio_value <= 0:
            return []

        month = ctx.timestamp.month
        in_winter = month in WINTER_MONTHS

        if in_winter:
            target_spy_shares = int(portfolio_value * self.spy_exposure / spy_price)
            target_tlt_shares = 0
        else:
            target_spy_shares = 0
            target_tlt_shares = int(portfolio_value * self.tlt_exposure / tlt_price)

        orders: list[Order] = []

        spy_pos = int(ctx.position("SPY").size)
        tlt_pos = int(ctx.position("TLT").size)

        # Sells first to free up cash before buys
        spy_delta = target_spy_shares - spy_pos
        tlt_delta = target_tlt_shares - tlt_pos

        # Issue sells first
        if spy_delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(spy_delta), symbol="SPY"))
        if tlt_delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(tlt_delta), symbol="TLT"))

        # Then buys
        if spy_delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(spy_delta), symbol="SPY"))
        if tlt_delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(tlt_delta), symbol="TLT"))

        return orders


UNIVERSE = ["SPY", "TLT"]

NAME = "halloween_sell_in_may"
HYPOTHESIS = (
    "Halloween / Sell-in-May seasonal: hold SPY from Nov 1 through Apr 30 (winter months) "
    "and rotate to TLT May-Oct (summer months); pure calendar seasonality exploiting "
    "well-documented equity risk premium asymmetry."
)

STRATEGY = HalloweenSellInMay()
