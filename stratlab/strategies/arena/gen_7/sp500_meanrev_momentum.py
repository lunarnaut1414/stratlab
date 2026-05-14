"""SP500 Mean-Reversion Within Momentum Trend — gen_7 sonnet-3

Hypothesis: hold top-20 SP500 stocks by negative 10d return (oversold dip)
that have positive 63d momentum (trend intact) and are above their 200d SMA;
equal-weight; VIX<30 gate; rebalance every 5 bars.

Rationale: Short-term mean reversion (buy recent dips) combined with
a longer-term momentum filter (only dip-buy stocks that are in uptrends).
The 200d SMA gate prevents buying into bear market downtrends. VIX<30
avoids panic selloffs which have different dynamics.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOMENTUM_WINDOW = 63      # ~3 months for trend filter
DIPS_WINDOW = 10          # short-term dip window
TREND_WINDOW = 200        # 200d SMA
TOP_K = 20
EXPOSURE = 0.97
VIX_THRESHOLD = 30.0


class Sp500MeanrevMomentum(Strategy):
    """Dip-buy SP500 stocks within uptrend: top-20 by negative 10d return
    with positive 63d momentum and above 200d SMA; VIX<30 gate.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        dips_window: int = DIPS_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        vix_threshold: float = VIX_THRESHOLD,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            dips_window=dips_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            vix_threshold=vix_threshold,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.dips_window = int(dips_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.vix_threshold = float(vix_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # VIX gate — avoid panic regimes
        vix_val = 20.0
        try:
            vix_hist = ctx.history("^VIX")
            if len(vix_hist) >= 2:
                vix_val = float(vix_hist["close"].iloc[-1])
        except KeyError:
            pass

        # SPY trend gate
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

        if not bull or vix_val >= self.vix_threshold:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            need = max(self.momentum_window, self.dips_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.dips_window + 2:
                return []

            # Score stocks: negative 10d return (dip) + positive 63d momentum
            scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # 200d SMA individual stock filter
                current_price = float(col.iloc[-1])
                if len(col) >= self.trend_window:
                    stock_sma = float(col.iloc[-self.trend_window:].mean())
                    if current_price <= stock_sma:
                        continue

                # 63d momentum must be positive
                p_start_63 = float(col.iloc[-self.momentum_window])
                if p_start_63 <= 0 or not np.isfinite(p_start_63):
                    continue
                mom_63 = current_price / p_start_63 - 1.0
                if not np.isfinite(mom_63) or mom_63 <= 0:
                    continue  # Must have positive trend

                # 10d dip: score by negative 10d return (most oversold = highest score)
                if len(col) < self.dips_window + 1:
                    continue
                p_start_10 = float(col.iloc[-self.dips_window])
                if p_start_10 <= 0 or not np.isfinite(p_start_10):
                    continue
                ret_10d = current_price / p_start_10 - 1.0
                if not np.isfinite(ret_10d):
                    continue

                # Score: negative 10d return (most dipped gets best rank)
                # scaled by momentum strength to prefer stronger-trend dips
                scores[sym] = -ret_10d * (1 + mom_63)

            if len(scores) < 5:
                # Not enough candidates — fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = weight

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
    return sp500_tickers() + ["TLT", "SPY", "^VIX"]


NAME = "sp500_meanrev_momentum"
HYPOTHESIS = (
    "SP500 mean-reversion with momentum filter: hold top-20 SP500 stocks by "
    "negative 10d return (oversold dip) that have positive 63d momentum (trend intact) "
    "and are above their 200d SMA; equal-weight; VIX<30 gate; rebalance every 5 bars"
)

UNIVERSE = _universe

STRATEGY = Sp500MeanrevMomentum()
