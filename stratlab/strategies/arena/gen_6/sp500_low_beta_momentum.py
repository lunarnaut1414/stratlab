"""SP500 low-beta momentum strategy.

Hypothesis: Rank SP500 stocks by the product of their 63-day total return
and the inverse of their 126-day realized beta to SPY. Hold top-20 equal-weight
when SPY is above its 200d SMA (bull market gate). Rotate to TLT when below.
Biweekly rebalance.

Rationale:
  The low-beta anomaly (Frazzini & Pedersen 2014) shows that low-beta stocks
  earn higher risk-adjusted returns than the CAPM predicts. Combining beta-
  adjustment with momentum selects stocks that are trending up strongly
  relative to the market AND are less sensitive to broad market moves. This
  creates a portfolio that participates in bull markets while exhibiting
  smaller drawdowns during corrections.

  The composite score (momentum * inverse_beta) differs fundamentally from:
  - Pure raw-return ranking (gen5_vix_gated_sp500_momentum)
  - Inverse-vol weighting (xsect_12m_invvol_goldencross — vol != beta)
  - 52-week high proximity (gen6_sp500_52wk_high_breakout)

Diversification vs leaderboard:
  - Beta measures systematic market sensitivity; vol measures total risk.
    Low-beta stocks may be high-vol (event-driven) while low-vol stocks may
    have high beta. The signals select different portfolios.
  - Monthly rebalance vs biweekly vs 10-bar — different trading rhythm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 63    # ~3 months return
BETA_WINDOW = 126       # ~6 months for beta estimation
REBALANCE_EVERY = 10    # biweekly
TOP_K = 20
TREND_WINDOW = 200      # SPY 200d SMA gate
EXPOSURE = 0.97
MIN_STOCKS = 10         # minimum qualifying stocks to take a position


class SP500LowBetaMomentum(Strategy):
    """Low-beta momentum: rank by 63d-return * (1/126d-beta), hold top-20."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            beta_window=beta_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.beta_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
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
            # Bull market: low-beta momentum ranking
            need = self.beta_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.beta_window:
                return []

            # SPY returns for beta calculation
            if "SPY" not in prices.columns:
                return []
            spy_returns = prices["SPY"].dropna().pct_change().dropna()
            if len(spy_returns) < self.beta_window - 2:
                return []
            spy_var = float(np.var(spy_returns.values))
            if spy_var <= 1e-8:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym == "SPY":
                    continue
                col = prices[sym].dropna()
                if len(col) < self.beta_window:
                    continue
                # 63d momentum
                if len(col) < self.momentum_window:
                    continue
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom) or mom <= 0:
                    continue
                # 126d beta vs SPY
                stock_returns = col.pct_change().dropna()
                # Align to same length as spy_returns
                min_len = min(len(stock_returns), len(spy_returns))
                if min_len < self.beta_window // 2:
                    continue
                sr = stock_returns.values[-min_len:]
                mr = spy_returns.values[-min_len:]
                cov = float(np.cov(sr, mr)[0, 1])
                beta = cov / spy_var
                if not np.isfinite(beta) or beta <= 0.01:
                    continue
                # Composite score: momentum * inverse_beta
                scores[sym] = mom / beta

            if len(scores) < MIN_STOCKS:
                # Not enough qualifying stocks; fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "sp500_low_beta_momentum"
HYPOTHESIS = (
    "SP500 low-beta momentum: rank SP500 stocks by product of 63d return and inverse "
    "126d realized beta to SPY, hold top-20 equal-weight when SPY above 200d SMA; "
    "TLT defensive; biweekly rebalance; beta-adjusted momentum distinct from raw-return or inv-vol ranking."
)

UNIVERSE = _universe

STRATEGY = SP500LowBetaMomentum()
