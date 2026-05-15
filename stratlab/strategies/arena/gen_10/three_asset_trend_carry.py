"""Three-asset trend-following with realized vol carry weighting.

Hypothesis: Hold SPY, TLT, and GLD each sized by their individual trend strength
(above/below moving average) and weighted inversely by realized vol. When all three
are in uptrend, hold all three proportionally. When some are in downtrend, reduce
their weight and reallocate to the trending assets. This is a pure trend-following /
volatility-carry approach without any macro signal gates.

Rationale:
  - Existing strategies all have BINARY regime switches (in/out of equity)
  - This approach maintains continuous allocation to all three asset classes
    weighted by their individual trend signals
  - Mechanically different from RSP breadth, credit spread, VIX, yield curve gates
  - Pure trend + vol-carry: similar to `gen6_rp_credit_tilt` but uses trend signals
    instead of static risk-parity + JNK overlay

Design:
  - SPY: trend_score = SPY 63d return / SPY 21d vol (Sharpe-like trend signal)
  - TLT: trend_score = TLT 63d return / TLT 21d vol
  - GLD: trend_score = GLD 63d return / GLD 21d vol
  - Weight each asset proportional to max(0, trend_score) — positive trend only
  - If no asset has positive trend: hold SHY
  - Final weights: normalized to sum to exposure (97%)
  - Rebalance every 10 bars
  - No explicit macro gate (self-gating through trend scores)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
TREND_WINDOW = 63         # momentum window
VOL_WINDOW = 21           # realized vol for Sharpe
EXPOSURE = 0.97
DEFENSIVE_ETF = "SHY"

ASSETS = ["SPY", "TLT", "GLD"]


class ThreeAssetTrendCarry(Strategy):
    """SPY/TLT/GLD trend-following with vol-carry weighting; SHY defensive;
    biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vol_window=vol_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vol_window + 10
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

        need = self.trend_window + self.vol_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < need - 5:
            return []

        trend_scores: dict[str, float] = {}

        for asset in ASSETS:
            if asset not in prices.columns:
                continue
            col = prices[asset].dropna()
            if len(col) < need - 5:
                continue

            arr = col.values

            # 63d momentum (Sharpe-like)
            if len(arr) < self.trend_window + 2:
                continue
            p_end = float(arr[-1])
            p_start = float(arr[-self.trend_window])
            if p_start <= 0:
                continue
            ret = p_end / p_start - 1.0

            # 21d realized vol
            if len(arr) < self.vol_window + 1:
                continue
            tail = arr[-(self.vol_window + 1):]
            logr = np.log(tail[1:] / tail[:-1])
            rv = float(np.std(logr))
            if rv <= 1e-6:
                continue

            # Sharpe-like score: normalize return by vol
            ann_ret = ret  # already annualizes via 63d window
            score = ann_ret / rv  # return per unit of vol
            trend_scores[asset] = score

        # Only use positive-trend assets
        positive_assets = {a: max(0.0, s) for a, s in trend_scores.items() if s > 0}

        target: dict[str, float] = {}

        if not positive_assets:
            # All assets in downtrend — defensive
            if DEFENSIVE_ETF in closes_now.index:
                target[DEFENSIVE_ETF] = self.exposure
        else:
            total_score = sum(positive_assets.values())
            if total_score <= 0:
                if DEFENSIVE_ETF in closes_now.index:
                    target[DEFENSIVE_ETF] = self.exposure
            else:
                for asset, score in positive_assets.items():
                    target[asset] = self.exposure * score / total_score

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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


NAME = "three_asset_trend_carry"
HYPOTHESIS = (
    "SPY/TLT/GLD three-asset trend-following with vol-carry weighting: compute 63d return / 21d "
    "realized-vol (Sharpe-like trend signal) for each asset; weight proportional to positive "
    "trend scores; hold SHY when all three are in downtrend; biweekly rebalance; no explicit "
    "macro gate — self-gating through trend scores; mechanically distinct from all SP500 stock "
    "pickers and binary macro-gating strategies on leaderboard"
)

UNIVERSE = ASSETS + [DEFENSIVE_ETF]

STRATEGY = ThreeAssetTrendCarry()
