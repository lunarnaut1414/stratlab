"""SP500 quality-momentum composite strategy.

Hypothesis: Combine momentum (63d return) with a "quality" proxy —
distance from 52-week high as a stability signal. High-quality
momentum stocks show both upward price trend AND price stability
(staying near their highs without wild swings). Stocks with high
momentum + low distance from 52wk-high = quality momentum.

Size positions inversely proportional to 20d realized volatility
(inverse-vol weighting). Gate on SPY 200d SMA. TLT defensive.
Biweekly rebalance (every 10 bars).

Key distinctions from gen6_sp500_52wk_high_breakout:
  - Scoring: pure momentum (63d return), no proximity filter threshold
  - Weighting: inverse-vol (not equal-weight)
  - Top-K: 15 vs 20
  - Fewer positions, more concentrated

Distinctions from curated xsect_12m_invvol_goldencross:
  - Shorter momentum window (63d vs 126d)
  - No skip window
  - SPY 200d SMA (same gate)
  - Different top-K (15 vs 20)
  - TLT defensive vs IEF
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly (~10 trading days)
MOMENTUM_WINDOW = 63     # 3-month return
VOL_WINDOW = 20          # 20d realized vol for inverse-vol weights
TOP_K = 15
TREND_WINDOW = 200       # SPY 200d SMA gate
EXPOSURE = 0.97
_HIGH_WINDOW = 126       # 6-month high for quality filter


class SP500QualityMomentum(Strategy):
    """Top SP500 stocks by 63d momentum, inverse-vol weighted,
    with SPY 200d SMA gate. TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
        high_window: int = _HIGH_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
            high_window=high_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self.high_window = int(high_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + self.vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
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
            # Bear market: TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            need = self.high_window + self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.high_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue
                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue

                # 3-month momentum
                if len(col) < self.momentum_window:
                    continue
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom) or mom <= 0:
                    continue

                # 6-month high proximity (quality proxy)
                rolling_high = float(col.iloc[-self.high_window:].max())
                if rolling_high <= 0:
                    continue
                proximity_to_high = current_price / rolling_high  # 1.0 = at high

                # Composite score: momentum weighted by proximity-to-high
                composite = mom * proximity_to_high
                if not np.isfinite(composite):
                    continue

                # Inverse realized volatility for sizing
                if len(col) < self.vol_window + 1:
                    continue
                tail = col.iloc[-self.vol_window - 1:]
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = composite
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
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


NAME = "sp500_quality_momentum"
HYPOTHESIS = (
    "SP500 quality-momentum tilt: hold top-15 SP500 stocks by composite score "
    "(63d momentum * 6m-high proximity ratio), inverse-vol weighted; SPY 200d "
    "SMA gate; TLT defensive; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SP500QualityMomentum()
