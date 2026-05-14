"""Commodity-equity regime rotation strategy.

Hypothesis: use DBC (commodity ETF) vs TLT (bond ETF) 42d momentum spread
as inflation-regime signal; when DBC outperforms TLT (inflation rising) hold
XLE+XLB+XLI equally (cyclical/commodity producers); when TLT leads hold
QQQ+XLK equally (deflation/growth); when both negative hold SHY; weekly rebalance.

Rationale: The DBC/TLT momentum spread captures the inflation regime:
  - DBC > TLT (commodities outperform bonds): inflation risk-on regime;
    commodity producers (energy XLE, materials XLB, industrials XLI) are
    the natural beneficiaries
  - TLT > DBC (bonds outperform commodities): deflation/disinflationary
    regime; growth tech stocks (QQQ, XLK) thrive as rates fall or stay low
  - Both negative: defensive cash-like SHY

This is distinct from VIX-based, credit-spread-based, or breadth-based
regime strategies already on the leaderboard. The DBC/TLT spread as a pure
inflation signal is not used by any existing strategy.

Distinction from existing strategies:
  - DBC vs TLT spread as inflation regime (entirely new signal)
  - Cyclical sector basket (XLE+XLB+XLI) vs deflation basket (QQQ+XLK)
  - Captures inflation/deflation cycle not captured by VIX or credit spreads
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
SPREAD_WINDOW = 42       # ~2 months
EXPOSURE = 0.97

# Signal ETFs (signal only via history - DBC and TLT are tradeable but used as signal)
# Tradeable holdings
INFLATION_BASKET = [("XLE", 1/3), ("XLB", 1/3), ("XLI", 1/3)]
DEFLATION_BASKET = [("QQQ", 0.5), ("XLK", 0.5)]
DEFENSIVE = [("SHY", 1.0)]


class CommodityEquityRegime(Strategy):
    """DBC/TLT inflation-regime rotation: cyclicals vs growth vs SHY.

    When DBC outperforms TLT on 42d momentum: hold XLE+XLB+XLI equally.
    When TLT outperforms DBC: hold QQQ+XLK equally.
    When both negative momentum: hold SHY.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spread_window: int = SPREAD_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spread_window=spread_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.spread_window = int(spread_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spread_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get DBC and TLT momentum
        dbc_ret = float("nan")
        tlt_ret = float("nan")

        try:
            dbc_hist = ctx.history("DBC")
            dbc_close = dbc_hist["close"].dropna()
            if len(dbc_close) >= self.spread_window:
                dbc_ret = float(dbc_close.iloc[-1] / dbc_close.iloc[-self.spread_window] - 1.0)
        except Exception:
            pass

        try:
            tlt_hist = ctx.history("TLT")
            tlt_close = tlt_hist["close"].dropna()
            if len(tlt_close) >= self.spread_window:
                tlt_ret = float(tlt_close.iloc[-1] / tlt_close.iloc[-self.spread_window] - 1.0)
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine regime
        target: dict[str, float] = {}

        if np.isnan(dbc_ret) or np.isnan(tlt_ret):
            # Fallback if data unavailable
            if "SHY" in closes_now.index:
                target["SHY"] = self.exposure
        elif dbc_ret < 0 and tlt_ret < 0:
            # Both negative: defensive cash-like
            if "SHY" in closes_now.index:
                target["SHY"] = self.exposure
        elif dbc_ret > tlt_ret:
            # Inflation regime: hold commodity/cyclical producers
            basket = INFLATION_BASKET
            total_w = sum(w for _, w in basket if f"{_}" in closes_now.index)
            available = [(s, w) for s, w in basket if s in closes_now.index]
            if available:
                total_weight = sum(w for _, w in available)
                for sym, w in available:
                    target[sym] = self.exposure * w / total_weight
            else:
                if "SHY" in closes_now.index:
                    target["SHY"] = self.exposure
        else:
            # Deflation/disinflation regime: hold growth/tech
            basket = DEFLATION_BASKET
            available = [(s, w) for s, w in basket if s in closes_now.index]
            if available:
                total_weight = sum(w for _, w in available)
                for sym, w in available:
                    target[sym] = self.exposure * w / total_weight
            else:
                if "SHY" in closes_now.index:
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


UNIVERSE = ["DBC", "TLT", "XLE", "XLB", "XLI", "QQQ", "XLK", "SHY"]

NAME = "commodity_equity_regime"
HYPOTHESIS = (
    "Commodity-equity regime rotation: use DBC (commodity ETF) vs TLT (bond ETF) "
    "42d momentum spread as inflation-regime signal; when DBC outperforms TLT "
    "(inflation rising) hold XLE+XLB+XLI equally (cyclical/commodity producers); "
    "when TLT leads hold QQQ+XLK equally (deflation growth); when both negative hold SHY; "
    "rebalance weekly"
)

STRATEGY = CommodityEquityRegime()
