"""Low-beta equal-weight SP500 momentum strategy.

Hypothesis: Rank SP500 stocks by 252d beta to SPY (ascending), hold top-20
lowest-beta stocks above their 100d SMA with positive 63d momentum; equal-
weight; SPY 200d SMA gate; IEF defensive; biweekly rebalance.

Rationale: The low-volatility/low-beta anomaly is one of the most robust
factors in finance. Combining beta-sorting with an uptrend filter (100d SMA)
and positive short-term momentum (63d) ensures we own low-beta stocks that
are actually working — not value traps. Equal-weight avoids concentration.
IEF (7-10yr treasury) is less correlated with stocks than TLT (20yr), giving
a different defensive path.

Distinction from existing strategies:
  - Uses SPY 252d beta as the primary ranking signal (no existing strategy does this)
  - 100d SMA per-stock filter (different from 50d/200d used elsewhere)
  - 63d momentum as a secondary confirmation filter
  - IEF defensive (instead of TLT/SHY)
  - Equal-weight
  - Biweekly rebalance (every 10 bars)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
BETA_WINDOW = 252          # 1-year beta estimation
TREND_WINDOW = 200         # SPY 200d SMA regime gate
STOCK_TREND_WINDOW = 100   # per-stock 100d SMA filter
MOMENTUM_WINDOW = 63       # 63d momentum confirmation
TOP_K = 20
EXPOSURE = 0.97


class LowBetaMomentumSP500(Strategy):
    """Low-beta SP500: top-20 lowest-beta stocks above 100d SMA with positive
    63d momentum; equal-weight; SPY 200d SMA gate; IEF defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        beta_window: int = BETA_WINDOW,
        trend_window: int = TREND_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            beta_window=beta_window,
            trend_window=trend_window,
            stock_trend_window=stock_trend_window,
            momentum_window=momentum_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.beta_window = int(beta_window)
        self.trend_window = int(trend_window)
        self.stock_trend_window = int(stock_trend_window)
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.beta_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA for market regime
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
            # Defensive: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Get price history for cross-section
            need = self.beta_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.beta_window:
                return []

            # Compute SPY returns for beta calculation
            spy_col = prices.get("SPY") if "SPY" in prices.columns else None
            if spy_col is None:
                return []
            spy_ret_series = spy_col.dropna().pct_change().dropna()
            if len(spy_ret_series) < self.beta_window - 5:
                return []
            spy_rets = spy_ret_series.iloc[-self.beta_window:].values
            spy_var = float(np.var(spy_rets))
            if spy_var <= 1e-10:
                return []

            betas: dict[str, float] = {}
            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.beta_window:
                    continue

                # Per-stock 100d SMA filter
                if len(col) < self.stock_trend_window:
                    continue
                stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                if float(col.iloc[-1]) < stock_sma:
                    continue  # Only stocks in uptrend

                # 63d momentum confirmation
                if len(col) < self.momentum_window + 2:
                    continue
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom) or mom <= 0:
                    continue  # Only positive momentum stocks

                # Compute beta to SPY
                stock_ret_series = col.pct_change().dropna()
                if len(stock_ret_series) < self.beta_window - 5:
                    continue

                # Align on same dates
                stock_rets = stock_ret_series.iloc[-self.beta_window:].values
                min_len = min(len(spy_rets), len(stock_rets))
                if min_len < 50:
                    continue
                s_rets = stock_rets[-min_len:]
                m_rets = spy_rets[-min_len:]
                cov = float(np.cov(s_rets, m_rets)[0, 1])
                beta = cov / spy_var
                if not np.isfinite(beta) or beta < 0:
                    continue  # skip negative-beta and NaN

                betas[sym] = beta

            if len(betas) < 5:
                # Not enough candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                # Sort ascending by beta (lowest beta first)
                k = min(self.top_k, len(betas))
                ranked = sorted(betas, key=betas.__getitem__)[:k]
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
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "lowbeta_momentum_sp500"
HYPOTHESIS = (
    "Low-beta equal-weight SP500 long: rank SP500 stocks by 252d beta to SPY (ascending), "
    "hold top-20 lowest-beta stocks above their 100d SMA with positive 63d momentum; "
    "equal-weight; SPY 200d SMA gate; IEF defensive; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = LowBetaMomentumSP500()
