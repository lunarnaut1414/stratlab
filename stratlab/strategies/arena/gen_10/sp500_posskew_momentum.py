"""SP500 momentum with per-stock 21d return-consistency filter.

Hypothesis (sonnet-9, gen_10):
    Return consistency measures how reliably a stock generates positive daily
    returns. A stock with 60%+ positive days over 21 bars is in a steady
    uptrend — not a burst momentum followed by reversal. Filtering to stocks
    with positive-day fraction >= 0.57 (12 of 21 days positive) selects names
    with durable daily upward pressure, orthogonal to RSI (oscillator-based) or
    near-high (price-position) quality screens.

    This filter is motivated by the observation that momentum crashes often
    happen when stocks have high 6-month returns driven by a few large-gap days,
    not consistent upward drift. The consistency filter weeds out spike-driven
    momentum in favor of steady-trending names.

    Design:
      - Compute fraction of positive daily returns over last 21 bars.
      - Only rank stocks with positive-day fraction >= 0.57.
      - Rank qualifying stocks by 126d momentum.
      - Hold top-15 inverse-vol weighted.
      - SPY 200d SMA outer bear gate to IEF.
      - Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen9_sp500_rsi_quality_momentum (OOS 0.88): RSI floor — level-based
    oscillator. A stock can have RSI > 35 with 40% positive days (it bounced
    from oversold but is inconsistent). Return consistency is different.
  - gen6_nearhi_momentum_quality: price vs 52w-high. A stock near its high
    can have gotten there via spike (one day +10%) not consistency.
  - No leaderboard strategy uses daily-return-consistency as a quality screen.

OOS resilience rationale:
  - Positive-day fraction filter is regime-invariant: consistent gainers exist
    in all VIX regimes. The filter doesn't depend on IS calm-VIX bias.
  - Selecting stocks with durable daily upward pressure avoids spike-driven
    momentum which tends to reverse, improving OOS persistence.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10           # biweekly
MOM_LOOKBACK = 126             # ~6 months
CONSISTENCY_WINDOW = 21        # 21-day window for daily-return consistency
CONSISTENCY_THRESHOLD = 0.57   # minimum fraction of positive days (12 of 21)
VOL_WINDOW_INDIV = 21          # for inverse-vol weights
SPY_TREND_WINDOW = 200         # outer gate
TOP_K = 15
EXPOSURE = 0.97
ANNUALIZATION = 252


def _compute_positive_day_fraction(prices: np.ndarray, window: int) -> float:
    """Compute fraction of positive daily log-returns over last `window` bars.

    Returns NaN if insufficient data.
    """
    if len(prices) < window + 1:
        return float("nan")
    tail = prices[-(window + 1):]
    logr = np.log(tail[1:] / tail[:-1])
    if len(logr) < window:
        return float("nan")
    logr = logr[-window:]
    pos_frac = float(np.sum(logr > 0)) / len(logr)
    return pos_frac


class SP500PosSkewMomentum(Strategy):
    """SP500 126d momentum with 21d return-consistency filter (positive-day fraction >= 0.57);
    inverse-vol weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + CONSISTENCY_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA outer gate
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
            need = MOM_LOOKBACK + CONSISTENCY_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + 2:
                    continue

                # 126d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_LOOKBACK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Return consistency filter: require >= 57% positive days in last 21d
                pos_frac = _compute_positive_day_fraction(col.values, CONSISTENCY_WINDOW)
                if not np.isfinite(pos_frac) or pos_frac < CONSISTENCY_THRESHOLD:
                    continue

                # Inverse-vol weight
                tail = col.values[-(VOL_WINDOW_INDIV + 1):]
                if len(tail) < VOL_WINDOW_INDIV + 1:
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

NAME = "sp500_posskew_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum filtered to stocks with 21d positive-day fraction >= 57%% "
    "(consistent daily upward pressure — avoids spike-driven momentum that reverses); "
    "inverse-vol weighted; SPY 200d SMA gate to IEF; biweekly rebalance — return-consistency "
    "filter selects names with durable daily uptrend not burst momentum, orthogonal to "
    "RSI/BB/ADX/near-high quality screens on leaderboard"
)

STRATEGY = SP500PosSkewMomentum()
