"""SP500 momentum with Bollinger Band squeeze quality filter.

Hypothesis (sonnet-10, gen_10):
    Rank SP500 stocks by 126d momentum, but apply a Bollinger Band squeeze
    filter: only include stocks where BB width (upper - lower) is BELOW the
    80th percentile of its own 63d BB-width distribution (compressed/squeezing)
    AND price is in the upper half of the band (%b > 0.5, positioned for
    breakout). This targets stocks building momentum without overextension.

Rationale:
  - BB squeeze (narrow bands) = low realized vol = energy building for a move.
  - Combined with %b > 0.5 (upper half), this selects stocks already trending
    up BUT with compressed volatility — the ideal profile for a breakout.
  - Different from the BB %b > 0.9 EXCLUSION already committed by sonnet-5
    (gen_10): that excludes overbought names; this SELECTS compressed-but-
    trending names (orthogonal filter — squeeze is about width, not position).
  - SPY 200d outer gate; inverse-vol weighted; biweekly rebalance.

Diversification:
  - No leaderboard entry uses BB squeeze as a stock quality filter.
  - The squeeze+direction combo selects different names from RSI, near-52w-high,
    MACD, or SMA-distance filters already on the leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_WINDOW = 126           # 6-month momentum
SPY_TREND_WINDOW = 200
BB_PERIOD = 20             # Bollinger Band period
BB_STD = 2.0               # Bollinger Band std multiplier
BB_SQUEEZE_WINDOW = 63     # lookback to determine width percentile
BB_SQUEEZE_THRESH = 80.0   # below 80th pct of own width = squeeze
BB_POSITION_FLOOR = 0.5    # %b must be > 0.5 (upper half of band)
TOP_K = 15
EXPOSURE = 0.97
VOL_WINDOW_INV = 21        # per-stock inverse-vol weighting


def _compute_bb_squeeze(col: "np.ndarray", bb_period: int, bb_std: float,
                         squeeze_window: int) -> tuple[float, float]:
    """Compute BB %b and squeeze percentile for the most recent bar.

    Returns (pct_b, squeeze_pct_rank):
      - pct_b: current position within band (0=lower, 1=upper, can exceed)
      - squeeze_pct_rank: pct rank of current BB width vs last squeeze_window
        bars of width values (0-100). Lower = tighter squeeze.
    """
    min_len = bb_period + squeeze_window + 2
    if len(col) < min_len:
        return float("nan"), float("nan")

    # Compute rolling BB widths over the squeeze window
    widths = []
    for i in range(squeeze_window + 1):
        end = len(col) - squeeze_window + i
        start = end - bb_period
        if start < 0:
            continue
        window = col[start:end]
        mid = float(np.mean(window))
        std = float(np.std(window))
        if std <= 0 or not np.isfinite(std):
            widths.append(float("nan"))
        else:
            widths.append(2 * bb_std * std / mid if mid > 0 else float("nan"))

    # Current width (last element) and its percentile vs prior
    if len(widths) < 2:
        return float("nan"), float("nan")

    current_width = widths[-1]
    prior_widths = [w for w in widths[:-1] if np.isfinite(w)]
    if not prior_widths or not np.isfinite(current_width):
        return float("nan"), float("nan")

    squeeze_pct = float(np.mean(np.array(prior_widths) < current_width)) * 100.0

    # Current %b
    end_idx = len(col)
    start_idx = end_idx - bb_period
    window = col[start_idx:end_idx]
    mid = float(np.mean(window))
    std = float(np.std(window))
    if std <= 0 or not np.isfinite(std) or not np.isfinite(mid):
        return float("nan"), squeeze_pct

    upper = mid + bb_std * std
    lower = mid - bb_std * std
    band_range = upper - lower
    if band_range <= 0:
        return float("nan"), squeeze_pct

    current_price = float(col[-1])
    pct_b = (current_price - lower) / band_range

    return pct_b, squeeze_pct


class SP500BBSqueezeMomentum(Strategy):
    """SP500 126d momentum with BB squeeze quality filter;
    inverse-vol weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOM_WINDOW, BB_PERIOD + BB_SQUEEZE_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE
        else:
            need = MOM_WINDOW + BB_PERIOD + BB_SQUEEZE_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOM_WINDOW + BB_SQUEEZE_WINDOW:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                col_arr = col.values

                # Minimum data requirement
                if len(col_arr) < MOM_WINDOW + BB_PERIOD + BB_SQUEEZE_WINDOW:
                    continue

                # 126d momentum
                p_end = float(col_arr[-1])
                p_start = float(col_arr[-MOM_WINDOW])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Bollinger Band squeeze filter
                pct_b, squeeze_pct = _compute_bb_squeeze(
                    col_arr, BB_PERIOD, BB_STD, BB_SQUEEZE_WINDOW
                )

                # Skip if we can't compute BB metrics
                if not np.isfinite(pct_b) or not np.isfinite(squeeze_pct):
                    continue

                # Must be in upper half of band (trending up, not breakdown)
                if pct_b < BB_POSITION_FLOOR:
                    continue

                # Must be in squeeze (bandwidth below 80th pct of recent width)
                if squeeze_pct >= BB_SQUEEZE_THRESH:
                    continue

                # Per-stock inverse-vol weight
                tail = col_arr[-(VOL_WINDOW_INV + 1):]
                if len(tail) < VOL_WINDOW_INV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = EXPOSURE * inv_vols[sym] / iv_sum

        # Build orders
        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "sp500_bb_squeeze_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum filtered to stocks with Bollinger Band squeeze "
    "(BB width below 80th pct of own 63d BB-width distribution) AND price in upper "
    "half of BB (%b > 0.5): squeeze identifies stocks building momentum without "
    "overextension; inverse-vol weighted; SPY 200d outer gate to IEF; biweekly "
    "rebalance — BB squeeze as compressed-energy quality filter is different from "
    "BB %b > 0.9 overbought exclusion already committed by other agents"
)

STRATEGY = SP500BBSqueezeMomentum()
