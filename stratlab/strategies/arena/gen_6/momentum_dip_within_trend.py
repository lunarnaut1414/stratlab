"""Short-term dip-buy within trend strategy.

Hypothesis: When SPY above 200d SMA (bull market), hold top-15 SP500 stocks
that (a) are in the top-30% of 63d momentum AND (b) have had a recent 5-day
pullback of at least -1.5% (buying momentum names on dips).
When fewer than 5 candidates pass both filters, fall back to top-20 by
63d momentum only (no dip requirement).
Rebalance every 5 bars. Defensive: TLT 60% + SHY 37% when SPY<200d SMA.

Structural distinctions vs existing leaderboard:
- Combines intermediate-term (63d) momentum with short-term (5d) reversal filter
- Actively requires recent pullback, not just highest momentum
- Different selection universe than pure momentum: favors oversold momentum names
- May generate different sector and timing exposure than near-52wk-high strategies
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 63
SHORT_WINDOW = 5
DIP_THRESHOLD = -0.015   # -1.5% minimum recent pullback
TOP_K = 15
FALLBACK_K = 20
MOMENTUM_PERCENTILE = 0.70   # must be in top-30% of momentum
REBALANCE_EVERY = 5
TREND_WINDOW = 200
EXPOSURE = 0.97
DEFENSIVE = {"TLT": 0.60, "SHY": 0.37}


class MomentumDipWithinTrend(Strategy):
    """Momentum dip-buy: top-30% 63d momentum with recent 5d pullback, SPY-gated."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        short_window: int = SHORT_WINDOW,
        dip_threshold: float = DIP_THRESHOLD,
        top_k: int = TOP_K,
        fallback_k: int = FALLBACK_K,
        momentum_percentile: float = MOMENTUM_PERCENTILE,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            short_window=short_window,
            dip_threshold=dip_threshold,
            top_k=top_k,
            fallback_k=fallback_k,
            momentum_percentile=momentum_percentile,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.short_window = int(short_window)
        self.dip_threshold = float(dip_threshold)
        self.top_k = int(top_k)
        self.fallback_k = int(fallback_k)
        self.momentum_percentile = float(momentum_percentile)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 5
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

        # SPY trend gate
        bull = True
        try:
            spy_hist = ctx.history("SPY")
            spy_c = spy_hist["close"].dropna()
            if len(spy_c) >= self.trend_window:
                bull = float(spy_c.iloc[-1]) > float(spy_c.iloc[-self.trend_window:].mean())
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull:
            for sym, wt in DEFENSIVE.items():
                if sym in live:
                    target[sym] = wt * self.exposure
        else:
            need = self.momentum_window + self.short_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + self.short_window:
                return []

            # Compute momentum and recent return for all stocks
            momentum_scores: dict[str, float] = {}
            short_returns: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.short_window:
                    continue
                # 63d momentum
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                # 5d return
                short_ret = float(col.iloc[-1] / col.iloc[-self.short_window] - 1.0)
                if np.isfinite(mom) and np.isfinite(short_ret):
                    momentum_scores[sym] = mom
                    short_returns[sym] = short_ret

            if not momentum_scores:
                return []

            # Compute momentum percentile threshold
            all_moms = sorted(momentum_scores.values())
            n = len(all_moms)
            threshold_idx = int(n * (1 - self.momentum_percentile))
            mom_threshold = all_moms[threshold_idx] if threshold_idx < n else float("-inf")

            # Filter: top momentum percentile AND recent dip
            dip_candidates = [
                sym for sym, mom in momentum_scores.items()
                if mom >= mom_threshold and short_returns.get(sym, 0) <= self.dip_threshold
            ]

            if len(dip_candidates) >= 5:
                # Sort by momentum and pick top_k
                dip_candidates.sort(key=lambda s: momentum_scores[s], reverse=True)
                longs = dip_candidates[: self.top_k]
            else:
                # Fallback: just pick top-fallback_k by momentum
                ranked = sorted(momentum_scores, key=momentum_scores.__getitem__, reverse=True)
                longs = ranked[: self.fallback_k]

            if not longs:
                return []

            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        orders: list[Order] = []

        # Sell positions not in target
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
    return sp500_tickers() + ["TLT", "SHY", "SPY"]


NAME = "momentum_dip_within_trend"
HYPOTHESIS = (
    "Short-term dip-buy within trend: when SPY above 200d SMA, hold top-15 SP500 stocks "
    "that are in top-30% 63d momentum quartile but have dropped >1.5% in the last 5 days "
    "(tactical dip); fallback to top-20 pure momentum if fewer than 5 pass; rebalance every 5 bars"
)
UNIVERSE = _universe
STRATEGY = MomentumDipWithinTrend()
