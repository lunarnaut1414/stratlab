"""SP500 momentum with volume expansion quality filter.

Hypothesis: Only hold SP500 stocks with EXPANDING trading volume (20d average
volume > 60d average volume) alongside strong 126d momentum. Volume expansion
signals institutional accumulation / increased conviction — a name in uptrend
with rising volume is more likely to continue than one with declining volume.
This filter is mechanically orthogonal to RSI, Bollinger Band position, and
MA-slope screens.

Rationale:
  - Price momentum on falling volume is fragile (distribution phase in Wyckoff analysis).
  - Price momentum on rising volume = institutional accumulation signal.
  - Volume expansion ratio as a quality screen has not appeared in any leaderboard
    strategy — it measures a different dimension of price action than pure price series.
  - Combining with per-stock 200d SMA trend gate adds a baseline uptrend requirement.

Design:
  - Compute 20d average volume and 60d average volume.
  - Require 20d avg_vol >= min_vol_ratio * 60d avg_vol (default 1.05 = 5% expansion).
  - Also require stock price > own 200d SMA.
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
VOL_SHORT_WINDOW = 20     # short-term volume average
VOL_LONG_WINDOW = 60      # long-term volume average (for expansion ratio)
MIN_VOL_RATIO = 1.05      # 20d avg vol must be >= 1.05x the 60d avg vol
STOCK_TREND_WINDOW = 200  # per-stock SMA gate
PRICE_VOL_WINDOW = 21     # for inverse-vol weight
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.50
VOL_TARGET = 0.13         # 13% annualized portfolio vol target
ANNUAL_FACTOR = 252.0


class SP500VolExpansionMomentum(Strategy):
    """SP500 126d momentum with volume-expansion quality filter; per-stock 200d SMA gate;
    inverse-vol weighted; portfolio vol-targeting; SPY 200d gate; IEF defensive;
    biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_short_window: int = VOL_SHORT_WINDOW,
        vol_long_window: int = VOL_LONG_WINDOW,
        min_vol_ratio: float = MIN_VOL_RATIO,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        price_vol_window: int = PRICE_VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
        vol_target: float = VOL_TARGET,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_short_window=vol_short_window,
            vol_long_window=vol_long_window,
            min_vol_ratio=min_vol_ratio,
            stock_trend_window=stock_trend_window,
            price_vol_window=price_vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
            vol_target=vol_target,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_short_window = int(vol_short_window)
        self.vol_long_window = int(vol_long_window)
        self.min_vol_ratio = float(min_vol_ratio)
        self.stock_trend_window = int(stock_trend_window)
        self.price_vol_window = int(price_vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)
        self.vol_target = float(vol_target)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.stock_trend_window, self.vol_long_window) + 10
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
            need = max(self.momentum_window, self.stock_trend_window, self.vol_long_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue

                # Need history with volume data
                try:
                    sym_hist = ctx.history(sym)
                except KeyError:
                    continue

                if len(sym_hist) < need - 5:
                    continue

                # Extract close prices
                if "close" not in sym_hist.columns:
                    continue
                close_col = sym_hist["close"].dropna()
                if len(close_col) < need - 10:
                    continue
                arr = close_col.values

                # Per-stock 200d SMA gate
                if len(arr) < self.stock_trend_window:
                    continue
                stock_sma = float(np.mean(arr[-self.stock_trend_window:]))
                if float(arr[-1]) <= stock_sma:
                    continue

                # Volume expansion filter
                if "volume" in sym_hist.columns:
                    vol_col = sym_hist["volume"].dropna()
                    vol_arr = vol_col.values
                    if len(vol_arr) >= self.vol_long_window:
                        short_avg = float(np.mean(vol_arr[-self.vol_short_window:]))
                        long_avg = float(np.mean(vol_arr[-self.vol_long_window:]))
                        if long_avg <= 0 or short_avg / long_avg < self.min_vol_ratio:
                            continue
                    # If not enough volume data, skip the filter (don't reject)

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
                if len(arr) < self.price_vol_window + 1:
                    continue
                tail = arr[-(self.price_vol_window + 1):]
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


NAME = "sp500_vol_expansion_momentum"
HYPOTHESIS = (
    "SP500 top-15 momentum (126d) with per-stock volume expansion filter: only hold stocks "
    "where 20d average volume >= 1.05x the 60d average volume (institutional accumulation signal); "
    "also require stock price above own 200d SMA; inverse-vol weighted; portfolio vol-target (13% ann); "
    "SPY 200d outer gate to IEF; biweekly rebalance — volume expansion identifies stocks with "
    "increasing institutional interest, not just price momentum, orthogonal to RSI/BB/MA-slope filters"
)

UNIVERSE = _universe

STRATEGY = SP500VolExpansionMomentum()
