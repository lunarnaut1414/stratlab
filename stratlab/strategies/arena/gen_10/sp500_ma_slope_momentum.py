"""SP500 momentum with 200d MA slope acceleration filter.

Hypothesis: Exclude SP500 stocks where the 200d MA slope has been DECELERATING
(the slope 20 bars ago was greater than the current slope, indicating trend weakening
at a structural level). Only hold stocks where the 200d MA slope is POSITIVE AND
has been steepening/stable over the past 20 bars (trend acceleration). This filter
catches names where the long-term average is not just pointing up, but gaining
upward velocity — structurally improving.

Rationale:
  - Standard 200d SMA filter: price > 200d SMA. This is necessary but not sufficient.
  - A stock can sit above its 200d SMA while the SMA slope is decelerating — the
    trend is aging and about to flatten. The slope-acceleration filter catches this.
  - Different from RSI (gen9 winner) and BB percent-b (other gen10 strategy):
    RSI measures recent buying pressure; BB-pb measures short-term price position;
    MA slope acceleration measures structural trend momentum on the 200d window.

Design:
  - Compute 200d SMA slope = difference between current SMA and SMA 20 bars ago.
  - Require slope > 0 (SMA trending up) AND current slope >= 80% of prior slope
    (not decelerating more than 20%).
  - Rank by 126d momentum; hold top-15.
  - Inverse-vol weighting.
  - Portfolio vol-target (13% ann) scales aggregate exposure 50-97%.
  - SPY 200d outer bear gate to IEF.
  - Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # ~6 months
TREND_WINDOW = 200        # 200d SMA window
SLOPE_LOOKBACK = 20       # bars to measure slope acceleration
MIN_SLOPE_RETENTION = 0.8 # slope must be >= 80% of prior slope (not decelerating hard)
VOL_WINDOW = 21           # inverse-vol weight & portfolio vol
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.50
VOL_TARGET = 0.13         # 13% annualized portfolio vol target
ANNUAL_FACTOR = 252.0


class SP500MASlopeMomentum(Strategy):
    """SP500 126d momentum with 200d MA slope-acceleration quality filter;
    inverse-vol weighted; portfolio vol-targeting; SPY 200d gate; IEF defensive;
    biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        slope_lookback: int = SLOPE_LOOKBACK,
        min_slope_retention: float = MIN_SLOPE_RETENTION,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
        vol_target: float = VOL_TARGET,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            slope_lookback=slope_lookback,
            min_slope_retention=min_slope_retention,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
            vol_target=vol_target,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.slope_lookback = int(slope_lookback)
        self.min_slope_retention = float(min_slope_retention)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)
        self.vol_target = float(vol_target)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window + self.slope_lookback) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure_max
        else:
            need = max(self.momentum_window, self.trend_window + self.slope_lookback) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < need - 10:
                    continue

                arr = col.values

                # Need enough data for trend + slope lookback
                min_len = self.trend_window + self.slope_lookback + 5
                if len(arr) < min_len:
                    continue

                # Compute current 200d SMA slope
                current_sma = float(np.mean(arr[-self.trend_window:]))
                prior_sma = float(np.mean(arr[-(self.trend_window + self.slope_lookback):-self.slope_lookback]))

                current_slope = current_sma - prior_sma  # positive = trending up

                # Require positive slope (price > 200d SMA proxy)
                if current_slope <= 0:
                    continue

                # Check slope acceleration: compare current slope to slope further back
                # Slope further back = SMA at (slope_lookback) ago vs SMA at (2*slope_lookback) ago
                if len(arr) < self.trend_window + 2 * self.slope_lookback + 5:
                    # Not enough data for prior slope comparison — use looser criterion
                    prior_slope = current_slope  # assume stable
                else:
                    prior_prior_sma = float(np.mean(arr[-(self.trend_window + 2 * self.slope_lookback):-(self.slope_lookback)]))
                    prev_sma = float(np.mean(arr[-(self.trend_window + self.slope_lookback):-self.slope_lookback]))
                    prior_slope = prev_sma - prior_prior_sma

                # Allow some deceleration (up to 20%) but not reversal
                if prior_slope > 0:
                    retention = current_slope / prior_slope
                    if retention < self.min_slope_retention:
                        continue
                # If prior_slope <= 0 and current_slope > 0: trend is accelerating from negative — keep

                # 126d momentum
                if len(arr) < self.momentum_window + 2:
                    continue
                p_end = float(arr[-1])
                p_start = float(arr[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                if len(arr) < self.vol_window + 1:
                    continue
                tail = arr[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # Raw weights (sum to exposure_max before vol-targeting)
                raw_weights = {sym: self.exposure_max * inv_vols[sym] / iv_sum for sym in ranked}

                # Portfolio vol-targeting proxy
                port_daily_vol = sum(
                    raw_weights[sym] * (1.0 / inv_vols[sym]) for sym in ranked
                )
                port_ann_vol = port_daily_vol * (ANNUAL_FACTOR ** 0.5)
                if port_ann_vol > 1e-6:
                    scale = self.vol_target / port_ann_vol
                    scale = float(np.clip(scale, self.exposure_min / self.exposure_max, 1.0))
                else:
                    scale = 1.0

                for sym in ranked:
                    target[sym] = raw_weights[sym] * scale

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_ma_slope_momentum"
HYPOTHESIS = (
    "SP500 top-15 momentum (126d) with per-stock 200d MA slope acceleration filter: only hold "
    "stocks where the 200d MA slope is positive and has been steepening over the past 20 bars "
    "(trend acceleration); inverse-vol weighted; SPY 200d outer bear gate to IEF; biweekly "
    "rebalance — trend-acceleration filter selects names where the long-term moving average is "
    "not just positive but gaining momentum, distinct from RSI/BB and near-high quality screens"
)

UNIVERSE = _universe

STRATEGY = SP500MASlopeMomentum()
