"""SP500 volatility-adjusted carry (Sharpe-ranked) strategy.

Hypothesis: rank SP500 stocks by (63d return / 21d realized vol) — a
Sharpe-like score — hold top-15, inverse-vol weighted; SPY 200d SMA gate;
GLD defensive when bear; rebalance every 15 bars.

Rationale: Pure return-momentum rewards stocks that moved a lot but may have
high volatility. The Sharpe-like score rewards stocks that achieved returns
with lower volatility — a return-per-unit-risk filter. This produces a
different population from pure momentum (nearhi, VIX-gated, etc.) because
a volatile stock with +30% return might score lower than a steady stock
with +15% return. The inverse-vol weighting further reduces concentration
in high-vol winners. GLD defensive (rather than TLT) provides inflation
protection and is distinct from TLT-defensive strategies.

Distinction from existing strategies:
  - Sharpe-like ranking score (return/vol) not used in any existing strategy
  - GLD as defensive ETF (not TLT or SHY) during bear regime
  - 15-bar rebalance period (distinct from 10-bar, 21-bar patterns)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 15    # ~3 weeks
MOMENTUM_WINDOW = 63    # ~3 months return
VOL_WINDOW = 21         # 1-month realized vol
TREND_WINDOW = 200      # SPY 200d SMA
TOP_K = 15
EXPOSURE = 0.97


class SharpeRankedSP500(Strategy):
    """SP500 Sharpe-ranked carry: top-15 SP500 stocks by (63d return / 21d vol),
    inverse-vol weighted; SPY 200d gate; GLD defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 10
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
            # Bear market: defensive GLD
            if "GLD" in closes_now.index:
                target["GLD"] = self.exposure
            elif "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: Sharpe-ranked stocks
            need = self.momentum_window + self.vol_window + 10
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + self.vol_window:
                return []

            sharpe_scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 5:
                    continue

                # 63d return
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # 21d realized vol
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                # Sharpe-like score: return per unit of volatility
                # Annualize vol for proper scaling
                ann_vol = rv * np.sqrt(252)
                if ann_vol <= 0:
                    continue
                sharpe_score = ret / ann_vol
                if not np.isfinite(sharpe_score):
                    continue

                # Only consider positive-return, positive-sharpe stocks
                if ret <= 0 or sharpe_score <= 0:
                    continue

                sharpe_scores[sym] = sharpe_score
                inv_vols[sym] = 1.0 / rv

            if len(sharpe_scores) < 5:
                # Not enough candidates — fall back to GLD
                if "GLD" in closes_now.index:
                    target["GLD"] = self.exposure
                elif "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(sharpe_scores))
                ranked = sorted(sharpe_scores, key=sharpe_scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["GLD", "TLT", "SPY"]


NAME = "sharpe_ranked_sp500"
HYPOTHESIS = (
    "SP500 volatility-adjusted carry: rank SP500 stocks by (63d return / 21d realized vol) "
    "— a Sharpe-like score — hold top-15, inverse-vol weighted; SPY 200d SMA gate; "
    "GLD defensive when bear (gold as alternative carry); rebalance every 15 bars; "
    "distinct from pure momentum or pure low-vol factor"
)

UNIVERSE = _universe

STRATEGY = SharpeRankedSP500()
