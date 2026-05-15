"""SP500 momentum with 200d-SMA-distance quality filter.

Hypothesis:
    Pure momentum strategies have two failure modes:
      1. Buying stocks that are BELOW their 200d SMA (downtrending broken stocks)
      2. Buying stocks that are EXTREMELY EXTENDED above their 200d SMA (parabolic
         names that revert hard when the trend breaks)

    Instead of a simple "above 200d SMA" gate, this strategy only considers stocks
    in a HEALTHY ZONE: price is between 100% and 130% of their 200d SMA (i.e.,
    0% to 30% above). This band:
      - Excludes broken stocks below SMA (same as most strategies)
      - Excludes parabolic overextension (>30% above SMA) which is new

    Within that healthy zone, rank by 126d momentum; hold top-15 inverse-vol weighted.
    SPY 200d outer market gate to IEF defensive.

Design:
    - Load closing prices for all SP500 stocks
    - Compute per-stock 200d SMA; compute price/SMA ratio
    - Only include stocks with ratio in [1.00, 1.30] (above SMA, not parabolic)
    - Rank by 126d total return; hold top-15
    - Inverse-vol weighting for position sizing
    - SPY 200d SMA outer gate: defensive IEF when SPY below SMA
    - Biweekly rebalance (every 10 bars)

Differentiation from leaderboard:
    - gen9_sp500_rsi_quality_momentum: uses RSI >= 35 quality filter, not SMA-distance
    - gen6_nearhi_momentum_quality: buys stocks NEAR 52w high (>0.80 ratio) — we
      EXCLUDE the most extended names; inverted direction
    - All other momentum strategies: simple "above 200d SMA" boolean, no upper cap
    - The SMA-distance band is a new dual-sided filter not present anywhere on the board
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly (~2 weeks)
MOMENTUM_WINDOW = 126      # 6-month momentum
SMA_WINDOW = 200           # per-stock and SPY 200d SMA
VOL_WINDOW = 21            # realized vol for inv-vol weights
TOP_K = 15                 # top stocks to hold
EXPOSURE = 0.97
SMA_LOW_RATIO = 1.00       # must be at least AT the 200d SMA
SMA_HIGH_RATIO = 1.30      # must not be more than 30% above SMA


class SP500SMADistanceMomentum(Strategy):
    """SP500 126d momentum with 200d-SMA-distance band filter (0-30% above SMA).

    Excludes both broken (below SMA) and parabolic (>30% above SMA) stocks.
    Inverse-vol weighted; SPY 200d gate to IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        sma_window: int = SMA_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        sma_low_ratio: float = SMA_LOW_RATIO,
        sma_high_ratio: float = SMA_HIGH_RATIO,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            sma_window=sma_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
            sma_low_ratio=sma_low_ratio,
            sma_high_ratio=sma_high_ratio,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.sma_window = int(sma_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.sma_low_ratio = float(sma_low_ratio)
        self.sma_high_ratio = float(sma_high_ratio)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sma_window + self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.sma_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.sma_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = self.sma_window + self.momentum_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.sma_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                n = len(col)

                # Need enough data for SMA + momentum
                if n < self.sma_window + 5:
                    continue

                # Per-stock 200d SMA distance filter
                sma_200 = float(col.iloc[-self.sma_window:].mean())
                if sma_200 <= 0:
                    continue
                current_price = float(col.iloc[-1])
                sma_ratio = current_price / sma_200
                # Only include stocks in healthy zone: 0% to 30% above SMA
                if sma_ratio < self.sma_low_ratio or sma_ratio > self.sma_high_ratio:
                    continue

                # 126d momentum
                if n < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Fallback to IEF when too few quality candidates
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target weights
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


NAME = "sp500_sma_distance_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with 200d-SMA-distance quality filter: "
    "exclude stocks >30% above their 200d SMA (parabolic extension) and below "
    "their 200d SMA; only rank stocks in healthy 0-30% above SMA zone; "
    "inverse-vol weighted; SPY 200d outer gate to IEF defensive; biweekly "
    "rebalance — SMA-distance band acts as simultaneous broken-stock and "
    "parabolic filter"
)

UNIVERSE = _universe

STRATEGY = SP500SMADistanceMomentum()
