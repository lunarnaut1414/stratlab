"""Cross-asset commodity carry rotation.

Hypothesis: Hold DBC (commodities) when DBC 20d return > SHY 20d return AND
GLD 20d momentum positive; hold TLT when equity (SPY) is below 150d SMA;
hold SHY otherwise; weekly rebalance.

Rationale: In inflation regimes, commodities outperform bonds and cash as a
carry trade. The DBC vs SHY return comparison tests whether commodities are
beating cash on a short horizon. GLD confirmation adds a second inflation-
signal layer. When equities are in a bear market (SPY below 150d SMA), bonds
(TLT) are the best risk-off asset. This 3-state model (commodity / bonds /
cash) is distinct from all existing strategies which focus on equity momentum.

Distinction from existing strategies:
  - Uses DBC (broad commodity ETF) as primary long vehicle
  - GLD confirmation as secondary inflation signal
  - 3-state: DBC / TLT / SHY — pure cross-asset carry
  - No equity stock picking involved
  - Different from copper_cycle_rotation: this is carry-based not MA-crossover
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
CARRY_WINDOW = 20          # 20-day return for carry comparison
TREND_WINDOW = 150         # SPY 150d SMA for equity regime
EXPOSURE = 0.97


class CommodityCarryRotation(Strategy):
    """Cross-asset commodity carry: DBC when commodity carry positive + GLD
    confirms; TLT in equity bear; SHY otherwise.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        carry_window: int = CARRY_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            carry_window=carry_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.carry_window = int(carry_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY trend regime ---
        spy_bear = False
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bear = float(spy_close.iloc[-1]) < spy_sma
        except KeyError:
            pass

        # --- DBC carry signal: DBC 20d return vs SHY 20d return ---
        dbc_carry_ok = False
        try:
            dbc_hist = ctx.history("DBC")
            shy_hist = ctx.history("SHY")
            if (len(dbc_hist) >= self.carry_window + 2 and
                    len(shy_hist) >= self.carry_window + 2):
                dbc_close = dbc_hist["close"].dropna()
                shy_close = shy_hist["close"].dropna()
                if (len(dbc_close) >= self.carry_window and
                        len(shy_close) >= self.carry_window):
                    dbc_ret = float(dbc_close.iloc[-1] / dbc_close.iloc[-self.carry_window] - 1.0)
                    shy_ret = float(shy_close.iloc[-1] / shy_close.iloc[-self.carry_window] - 1.0)
                    dbc_carry_ok = np.isfinite(dbc_ret) and np.isfinite(shy_ret) and dbc_ret > shy_ret
        except KeyError:
            pass

        # --- GLD momentum confirmation: GLD 20d return positive ---
        gld_positive = False
        try:
            gld_hist = ctx.history("GLD")
            if len(gld_hist) >= self.carry_window + 2:
                gld_close = gld_hist["close"].dropna()
                if len(gld_close) >= self.carry_window:
                    gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-self.carry_window] - 1.0)
                    gld_positive = np.isfinite(gld_ret) and gld_ret > 0.0
        except KeyError:
            pass

        # Determine target allocation
        target: dict[str, float] = {}

        if spy_bear:
            # Equity bear: safe haven bonds
            target["TLT"] = self.exposure
        elif dbc_carry_ok and gld_positive:
            # Commodity carry positive with inflation confirmation
            target["DBC"] = self.exposure
        else:
            # Neither signal fires: hold cash proxy
            target["SHY"] = self.exposure

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


UNIVERSE = ["DBC", "GLD", "TLT", "SHY", "SPY"]

NAME = "commodity_carry_rotation"
HYPOTHESIS = (
    "Cross-asset carry: hold DBC (commodities) when DBC 20d return > SHY 20d return AND "
    "GLD 20d momentum both positive; hold TLT when equity (SPY) is below 150d SMA; "
    "hold SHY otherwise; weekly rebalance; captures commodity carry in inflation regimes"
)

STRATEGY = CommodityCarryRotation()
