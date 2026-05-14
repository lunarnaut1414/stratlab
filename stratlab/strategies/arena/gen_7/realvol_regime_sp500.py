"""Realized-vol regime SP500 momentum strategy.

Hypothesis: when SPY 10d realized vol is below its 60d median (calm market),
hold top-15 SP500 stocks by 42d momentum equal-weight; when above median hold
SPY 50% + TLT 47%; rebalance every 5 bars.

Rationale: Unlike VIX (implied vol), realized volatility of SPY itself captures
the actual turbulence in the market microstructure, not fear premiums. When the
market is calm (realized vol below its recent median), momentum works well.
When turbulence rises (realized vol above median), momentum degrades and we
retreat to a balanced SPY+TLT allocation rather than full defensive. This is
different from VIX-level gating (which uses implied not realized vol) and from
credit spread gating.

Key distinctions:
  - Uses SPY realized vol vs its own 60d median — no VIX or credit signal
  - Partial retreat (SPY 50%+TLT 47%) not full defensive — stays in equities
  - 5-bar rebalance generates high trade count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # bars
MOMENTUM_WINDOW = 42      # ~2 months
REALVOL_SHORT = 10        # 10d realized vol window
REALVOL_LONG = 60         # 60d lookback for median baseline
TOP_K = 15
TREND_WINDOW = 200
EXPOSURE = 0.97


class RealVolRegimeSP500(Strategy):
    """Realized-vol regime SP500: calm=top-15 momentum, turbulent=SPY+TLT mix."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        realvol_short: int = REALVOL_SHORT,
        realvol_long: int = REALVOL_LONG,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            realvol_short=realvol_short,
            realvol_long=realvol_long,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.realvol_short = int(realvol_short)
        self.realvol_long = int(realvol_long)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.realvol_long + self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get SPY history for regime signal
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window + self.realvol_long + 5:
            return []

        # SPY 200d SMA gate
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: go defensive TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Compute 10d realized vol as annualized stddev of log-returns
            tail_prices = spy_close.iloc[-(self.realvol_short + 1):]
            logr_short = np.log(tail_prices.values[1:] / tail_prices.values[:-1])
            rv_short = float(np.std(logr_short)) * np.sqrt(252)  # annualized

            # Compute rolling 10d realized vol over last 60 bars
            rv_series = []
            for i in range(self.realvol_long):
                start = -(self.realvol_long + self.realvol_short) + i
                end = -(self.realvol_long) + i
                if end == 0:
                    window_prices = spy_close.iloc[start:]
                else:
                    window_prices = spy_close.iloc[start:end]
                if len(window_prices) < self.realvol_short + 1:
                    continue
                lr = np.log(window_prices.values[1:] / window_prices.values[:-1])
                rv_series.append(float(np.std(lr)) * np.sqrt(252))

            if len(rv_series) < 20:
                # Not enough history for median — default to momentum only
                calm = True
            else:
                rv_median = float(np.median(rv_series))
                calm = rv_short <= rv_median

            if calm:
                # Calm regime: hold top-K momentum stocks
                prices = ctx.closes_window(self.momentum_window + 5)
                if len(prices) < self.momentum_window:
                    return []

                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    # Fallback to SPY+TLT
                    if "SPY" in closes_now.index:
                        target["SPY"] = 0.50 * self.exposure
                    if "TLT" in closes_now.index:
                        target["TLT"] = 0.47 * self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
                    per_wt = self.exposure / len(longs)
                    for sym in longs:
                        target[sym] = per_wt
            else:
                # Turbulent regime: partial retreat — SPY 50% + TLT 47%
                if "SPY" in closes_now.index:
                    target["SPY"] = 0.50 * self.exposure
                if "TLT" in closes_now.index:
                    target["TLT"] = 0.47 * self.exposure

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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "realvol_regime_sp500"
HYPOTHESIS = (
    "Weekly VIX realized-vol spread regime: when SPY 10d realized vol is below its 60d median "
    "(calm market) hold top-15 SP500 stocks by 42d momentum equal-weight; when above median hold "
    "SPY 50%+TLT 47%; rebalance every 5 bars; realized-vol spread signal distinct from VIX level or credit gates"
)

UNIVERSE = _universe

STRATEGY = RealVolRegimeSP500()
