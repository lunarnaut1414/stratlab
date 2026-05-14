"""DBC commodity trend with inflation-regime sector rotation.

Hypothesis: Use DBC (broad commodity ETF) 60d SMA trend as an inflation-regime
signal to rotate between cyclical and defensive sector baskets:
  - Commodity uptrend (DBC > 60d SMA) AND SPY > 200d SMA: hold XLE+XLB+XLI
    equally (commodity-sensitive cyclical sectors)
  - Commodity downtrend (DBC < 60d SMA) AND SPY > 200d SMA: hold XLU+XLP+TLT
    equally (defensive sectors + bonds)
  - SPY bear (SPY < 200d SMA): hold SHY 97% (cash proxy)

Rationale: Commodity trends drive sector relative performance significantly — in
commodity uptrends, energy/materials/industrials outperform; in downtrends,
utilities/staples lead as the economy slows. This is structurally different from
VIX-gated, credit-spread, or breadth-signal strategies — it uses physical
commodity price trends as the primary macro signal.

Distinct from:
  - gen5_copper_cycle_rotation (uses CPER not DBC, different defensive basket)
  - All credit-spread and VIX-based strategies on the leaderboard
  - gen6 breadth-signal strategies (RSP, IWM)

Weekly rebalance (5 bars) generates sufficient trade count over IS window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
TREND_WINDOW = 200         # SPY trend filter
COMMODITY_MA = 60          # DBC SMA window
EXPOSURE = 0.97

_DBC = "DBC"
_SPY = "SPY"

# Cyclical sector basket (commodity uptrend)
CYCLICALS = ["XLE", "XLB", "XLI"]
# Defensive basket (commodity downtrend)
DEFENSIVES = ["XLU", "XLP", "TLT"]
# Bear market proxy
CASH_PROXY = "SHY"


class DBCCommoditySectorRotation(Strategy):
    """DBC commodity-trend inflation-regime sector rotation.

    Long-only: cyclicals when commodity uptrend + bull; defensives when
    commodity downtrend + bull; SHY when equity bear.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        commodity_ma: int = COMMODITY_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            commodity_ma=commodity_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.commodity_ma = int(commodity_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.commodity_ma) + 10
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

        # Determine SPY regime (bull vs bear)
        spy_bull = False
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # Determine DBC commodity trend
        commodity_up = False
        try:
            dbc_hist = ctx.history(_DBC)
            if dbc_hist is not None and len(dbc_hist) >= self.commodity_ma:
                dbc_close = dbc_hist["close"].dropna()
                if len(dbc_close) >= self.commodity_ma:
                    dbc_sma = float(dbc_close.iloc[-self.commodity_ma:].mean())
                    commodity_up = float(dbc_close.iloc[-1]) > dbc_sma
        except Exception:
            pass

        # Determine target allocation
        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — hold cash proxy
            if CASH_PROXY in live:
                target[CASH_PROXY] = self.exposure
        elif commodity_up:
            # Commodity uptrend — cyclical sectors
            available = [s for s in CYCLICALS if s in live]
            if available:
                per_weight = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_weight
        else:
            # Commodity downtrend — defensive sectors
            available = [s for s in DEFENSIVES if s in live]
            if available:
                per_weight = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_weight

        # Build orders
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


NAME = "dbc_commodity_sector_rotation"
HYPOTHESIS = (
    "DBC commodity trend with inflation-regime sector rotation: hold XLE+XLB+XLI equally "
    "when DBC above 60d SMA (commodity uptrend) AND SPY above 200d SMA; hold XLU+XLP+TLT "
    "when commodity downtrend; SHY when SPY bear; weekly rebalance; commodity-trend primary "
    "signal with sector tilt"
)

UNIVERSE = ["DBC", "SPY", "XLE", "XLB", "XLI", "XLU", "XLP", "TLT", "SHY"]

STRATEGY = DBCCommoditySectorRotation()
