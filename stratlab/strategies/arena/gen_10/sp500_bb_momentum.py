"""SP500 momentum with Bollinger-Band position (percent-b) filter.

Hypothesis: Exclude SP500 stocks with BB percent-b > 0.9 (extremely overbought
in short-term price channel) before applying 126d momentum ranking. This avoids
chasing extended breakout names that are statistically overextended relative to
their recent price band — they often mean-revert sharply. The filter is orthogonal
to RSI (gen9's best performer): RSI measures recent buying pressure, BB percent-b
measures current price position in a rolling band.

Design:
  - Compute Bollinger Band percent-b = (price - lower_band) / (upper_band - lower_band)
    using 20d window and 2 std dev.
  - Exclude stocks with percent-b > 0.9 (top decile of band extension).
  - Also require stock price > own 200d SMA (individual trend gate).
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
BB_WINDOW = 20            # Bollinger Band lookback
BB_STD = 2.0              # band width
BB_MAX = 0.9              # exclude percent-b > this
STOCK_TREND_WINDOW = 200  # per-stock SMA gate
VOL_WINDOW = 21           # inverse-vol weight & portfolio vol
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.50
VOL_TARGET = 0.13         # 13% annualized portfolio vol target
ANNUAL_FACTOR = 252.0


def _bb_pctb(prices: "np.ndarray", window: int, n_std: float) -> float:
    """Compute Bollinger Band percent-b for the most recent price.

    percent-b = (last_price - lower_band) / (upper_band - lower_band)
    Returns NaN if insufficient data.
    """
    if len(prices) < window:
        return float("nan")
    tail = prices[-window:]
    mid = float(np.mean(tail))
    std = float(np.std(tail, ddof=1))
    if std < 1e-10:
        return float("nan")
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_width = upper - lower
    if band_width < 1e-10:
        return float("nan")
    last = float(prices[-1])
    return (last - lower) / band_width


class SP500BBMomentum(Strategy):
    """SP500 126d momentum with BB percent-b exclusion filter; inverse-vol weighted;
    portfolio vol-targeting; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        bb_window: int = BB_WINDOW,
        bb_std: float = BB_STD,
        bb_max: float = BB_MAX,
        stock_trend_window: int = STOCK_TREND_WINDOW,
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
            bb_window=bb_window,
            bb_std=bb_std,
            bb_max=bb_max,
            stock_trend_window=stock_trend_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
            vol_target=vol_target,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.bb_window = int(bb_window)
        self.bb_std = float(bb_std)
        self.bb_max = float(bb_max)
        self.stock_trend_window = int(stock_trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)
        self.vol_target = float(vol_target)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.stock_trend_window, self.bb_window) + 10
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
            need = max(self.momentum_window, self.stock_trend_window, self.bb_window) + 5
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

                # Per-stock 200d SMA gate
                if len(arr) < self.stock_trend_window:
                    continue
                stock_sma = float(np.mean(arr[-self.stock_trend_window:]))
                if float(arr[-1]) <= stock_sma:
                    continue

                # Bollinger Band percent-b filter
                pctb = _bb_pctb(arr, self.bb_window, self.bb_std)
                if not np.isfinite(pctb) or pctb > self.bb_max:
                    continue

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

                # Compute raw weights (sum to exposure_max before vol-targeting)
                raw_weights = {sym: self.exposure_max * inv_vols[sym] / iv_sum for sym in ranked}

                # Portfolio vol-targeting: estimate portfolio daily vol from inverse-vol weights
                # proxy: weighted average of individual vols gives portfolio vol approximation
                # (ignores correlations — underestimates slightly, but mechanically robust)
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


NAME = "sp500_bb_momentum"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock Bollinger-Band position filter: exclude stocks with "
    "BB-percent-b > 0.9 (overbought extreme) before ranking; hold top-15 inverse-vol weighted "
    "above their own 200d SMA; portfolio vol-target (13% ann) scales aggregate exposure 50-97%; "
    "SPY 200d outer bear gate to IEF; biweekly rebalance — BB-position filter avoids chasing "
    "extended breakouts inside momentum ranking, orthogonal to RSI quality screen already on leaderboard"
)

UNIVERSE = _universe

STRATEGY = SP500BBMomentum()
