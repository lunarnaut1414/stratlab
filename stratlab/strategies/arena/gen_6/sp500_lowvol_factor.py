"""SP500 low-volatility factor strategy.

Hypothesis: Rank SP500 stocks by 63d realized volatility (ascending), hold
top-20 lowest-volatility stocks above their 200d SMA; equal-weight;
SPY 200d SMA gate; TLT defensive; monthly rebalance (21 bars).

Rationale: The low-volatility anomaly is one of the best-documented market
anomalies — low-vol stocks outperform on a risk-adjusted basis. Using 63d
realized vol as the sole ranking signal with a 200d SMA filter ensures we
pick defensive low-vol stocks that are in uptrends. Monthly rebalance keeps
turnover low. This is purely factor-based, not momentum-based.

Distinction from existing strategies:
  - Ranks by volatility ASCENDING (lowest first) — no momentum component
  - Only 200d SMA per-stock filter, not combined with momentum
  - Monthly rebalance (21 bars) — lower turnover than biweekly
  - No VIX or credit gate
  - Pure low-vol factor: very different daily return path from momentum
  - Different from gen6_lowbeta_momentum_sp500: no beta calculation,
    no momentum filter, different ranking signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
VOL_WINDOW = 63            # 63d realized vol for ranking
STOCK_TREND_WINDOW = 200   # per-stock 200d SMA uptrend filter
TREND_WINDOW = 200         # SPY 200d SMA market regime gate
TOP_K = 20
EXPOSURE = 0.97


class SP500LowVolFactor(Strategy):
    """SP500 low-volatility factor: top-20 lowest-vol stocks above 200d SMA;
    SPY 200d SMA gate; TLT defensive; monthly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            stock_trend_window=stock_trend_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.stock_trend_window = int(stock_trend_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vol_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA for market regime gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
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
            # Bear market: hold TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: low-vol factor
            need = self.stock_trend_window + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.stock_trend_window:
                return []

            vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "TLT"):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.stock_trend_window:
                    continue

                # Per-stock 200d SMA uptrend filter
                stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                if float(col.iloc[-1]) < stock_sma:
                    continue  # Only uptrending stocks

                # Compute 63d realized volatility
                if len(col) < self.vol_window + 2:
                    continue
                vol_tail = col.iloc[-self.vol_window - 1:]
                logr = np.log(vol_tail.values[1:] / vol_tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                vols[sym] = rv

            if len(vols) < 5:
                # Not enough candidates — fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Sort ascending by volatility (lowest vol first)
                k = min(self.top_k, len(vols))
                ranked = sorted(vols, key=vols.__getitem__)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

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


NAME = "sp500_lowvol_factor"
HYPOTHESIS = (
    "SP500 low-turnover low-vol factor: rank SP500 stocks by 63d realized volatility ascending, "
    "hold top-20 lowest-vol stocks above 200d SMA; equal-weight; SPY 200d SMA gate; "
    "TLT defensive; monthly rebalance (21 bars); pure low-vol factor without beta calculation"
)

UNIVERSE = _universe

STRATEGY = SP500LowVolFactor()
