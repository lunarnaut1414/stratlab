"""Commodity-equity divergence rotation strategy.

Hypothesis: When commodities (DBC) outperform equities (SPY) on 20d return,
it signals an inflation/commodity regime. Route to energy/materials/gold.
When equities lead, route to top-5 SP500 momentum stocks. SPY 200d SMA gate.

Rationale: The DBC/SPY 20d return spread captures macro regime shifts between
growth (equity-led) and inflation (commodity-led) phases. In commodity-led
regimes, energy, materials, and gold outperform. In equity-led growth regimes,
momentum stocks excel. This cross-asset divergence signal is fundamentally
different from vol-regime, credit-spread, and gold-miner signals already used.

Distinction from existing strategies:
  - DBC vs SPY relative return (not DBC vs SHY/cash carry) as regime signal
  - Routes to commodity-sensitive sectors (XLE+XLB) or momentum SP500 stocks
  - Biweekly rebalance, top-5 concentrated stock holding
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
DIV_WINDOW = 20         # 20d for commodity vs equity divergence
STOCK_MOM_WINDOW = 63   # 63d for stock momentum ranking
TOP_K = 5               # concentrated top-5 stock portfolio
TREND_WINDOW = 200
EXPOSURE = 0.97

COMMODITY_REGIME_ASSETS = ["XLE", "XLB", "GLD"]  # 1/3 each
DEFENSIVE = "TLT"


class CommodityEquityDivergence(Strategy):
    """DBC/SPY 20d divergence: commodity regime -> XLE+XLB+GLD; equity regime -> SP500 momentum."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        div_window: int = DIV_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            div_window=div_window,
            stock_mom_window=stock_mom_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.div_window = int(div_window)
        self.stock_mom_window = int(stock_mom_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_mom_window) + 10
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

        # SPY 200d SMA bear market gate
        bull_market = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    bull_market = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # Compute DBC vs SPY divergence
        commodity_regime = False
        try:
            dbc_hist = ctx.history("DBC")
            spy_hist = ctx.history("SPY")
            if (dbc_hist is not None and spy_hist is not None
                    and len(dbc_hist) >= self.div_window
                    and len(spy_hist) >= self.div_window):
                dbc_close = dbc_hist["close"].dropna()
                spy_close = spy_hist["close"].dropna()
                if (len(dbc_close) >= self.div_window
                        and len(spy_close) >= self.div_window):
                    dbc_ret = float(dbc_close.iloc[-1] / dbc_close.iloc[-self.div_window] - 1.0)
                    spy_ret_20 = float(spy_close.iloc[-1] / spy_close.iloc[-self.div_window] - 1.0)
                    commodity_regime = dbc_ret > spy_ret_20
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull_market:
            # Bear market: defensive TLT
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        elif commodity_regime:
            # Commodity regime: XLE + XLB + GLD equal-weight
            available = [s for s in COMMODITY_REGIME_ASSETS if s in closes_now.index]
            if available:
                per_w = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_w
            else:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
        else:
            # Equity regime: top-K SP500 momentum stocks
            need = self.stock_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.stock_mom_window:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.stock_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.stock_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
                        target[sym] = per_w

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["DBC", "GLD", "XLE", "XLB", "TLT", "SPY"]


NAME = "commodity_equity_divergence"
HYPOTHESIS = (
    "Commodity-equity divergence rotation: when DBC 20d return minus SPY 20d return is positive "
    "(commodities outperforming equities) hold XLE+XLB+GLD equal-weight; "
    "when equity regime (SPY 20d > DBC 20d) hold top-5 SP500 stocks by 63d momentum; "
    "SPY 200d SMA gate; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = CommodityEquityDivergence()
