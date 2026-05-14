"""SP500 Idiosyncratic Momentum — gen_7 sonnet-7

Hypothesis: Rank SP500 stocks by their idiosyncratic 63d return — i.e., the
stock's raw 63d return minus (63d beta * SPY 63d return). This selects stocks
that are outperforming the market on a risk-adjusted basis, not just riding the
broad market tide.

Rationale: Pure price momentum selects stocks with high beta in bull markets.
Idiosyncratic momentum (alpha relative to market) filters out market-beta-driven
winners, targeting stocks with genuine company-specific catalysts. This should
produce a portfolio distinct from raw-momentum strategies even during bull markets.

SPY 200d SMA gate: avoid holding individual stocks in bear regimes.
TLT as defensive fallback.
Biweekly rebalance: 10 bars for high trade count over IS window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # ~3 months
BETA_WINDOW = 126          # 6 months for beta estimation
TREND_WINDOW = 200         # 200d SMA gate
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"


class SP500IdiosyncraticMomentum(Strategy):
    """Idiosyncratic (market-adjusted) momentum on SP500 stocks."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            beta_window=beta_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.beta_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute idiosyncratic momentum
            need = max(self.beta_window, self.momentum_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                return []

            # SPY returns for beta computation
            if _SPY not in prices.columns:
                return []
            spy_prices = prices[_SPY].dropna()
            if len(spy_prices) < self.beta_window:
                return []

            # Use last beta_window of SPY returns
            spy_log_rets = np.log(spy_prices.values[1:] / spy_prices.values[:-1])
            spy_mom_ret = float(spy_prices.iloc[-1] / spy_prices.iloc[-self.momentum_window] - 1.0)

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym == _SPY or sym == _TLT:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.beta_window:
                    continue

                # Compute beta via covariance ratio over beta_window
                stock_log_rets = np.log(col.values[1:] / col.values[:-1])
                n = min(len(stock_log_rets), len(spy_log_rets))
                if n < 30:
                    continue
                stock_r = stock_log_rets[-n:]
                spy_r = spy_log_rets[-n:]
                if np.std(spy_r) < 1e-8:
                    continue
                beta = float(np.cov(stock_r, spy_r)[0, 1] / np.var(spy_r))
                if not np.isfinite(beta):
                    continue

                # 63d raw momentum
                if len(col) < self.momentum_window + 1:
                    continue
                raw_ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(raw_ret):
                    continue

                # Idiosyncratic return = raw - beta * market
                idio_ret = raw_ret - beta * spy_mom_ret
                if np.isfinite(idio_ret):
                    scores[sym] = idio_ret

            if len(scores) < 5:
                # Not enough candidates - fall back to TLT
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
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
    return sp500_tickers() + [_TLT, _SPY]


NAME = "sp500_idiosyncratic_momentum"
HYPOTHESIS = (
    "SP500 idiosyncratic momentum: rank SP500 stocks by 63d return minus beta-adjusted "
    "SPY return (residual alpha), hold top-15 above 200d SMA; equal-weight; TLT defensive "
    "in bear; biweekly rebalance; selects stocks outperforming market on risk-adjusted basis"
)

UNIVERSE = _universe

STRATEGY = SP500IdiosyncraticMomentum()
