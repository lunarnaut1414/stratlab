"""SP500 Momentum with Per-Stock Max-Drawdown Quality Screen — gen_10 sonnet-6

Hypothesis: Exclude SP500 stocks with max-drawdown > 12% over the last 21 days
before applying 126d momentum ranking. This filters out stocks in active breakdown
(falling knives) even if their medium-term momentum is still positive.

Rationale:
  - RSI quality filter (gen9 best performer at 96% OOS retention) uses a 14d
    oscillator. Max-drawdown filter is a different quality lens: it measures how
    much a stock has dropped from its recent peak, capturing breakdown dynamics
    more directly.
  - A stock with 126d momentum of +30% but -15% over 21d is in active distribution —
    the max-drawdown screen prevents buying into that breakdown.
  - Unlike RSI (centered around 50 in momentum names), max-drawdown has a clear
    structural interpretation: how far from the recent peak has the stock fallen.
  - Distinct from near-52w-high filter (long-window 252d) — this is a short-term
    21d protection against recent breakdown within a medium-term winner.

Design:
  - Per-stock quality filter: max-drawdown over 21d <= 12% (price drop from 21d peak).
  - Rank by 126d momentum.
  - Hold top-15 above SPY 200d SMA; inverse-vol weighted.
  - IEF defensive when SPY below 200d SMA.
  - Biweekly rebalance (every 10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 126      # ~6 months
DD_WINDOW = 21             # max-drawdown lookback (1 month)
MAX_DRAWDOWN_FLOOR = 0.12  # reject if dropped >12% from 21d peak
VOL_WINDOW = 21            # for inverse-vol weights
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97


def _compute_max_drawdown(prices: "np.ndarray") -> float:
    """Compute max peak-to-trough drawdown from a 1d price array.

    Returns a positive fraction (e.g. 0.15 = 15% drawdown).
    """
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for p in prices[1:]:
        if p > peak:
            peak = p
        dd = (peak - p) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


class SP500MaxDDQualityMomentum(Strategy):
    """SP500 top-15 by 126d momentum with 21d max-drawdown quality screen;
    inverse-vol weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        dd_window: int = DD_WINDOW,
        max_drawdown_floor: float = MAX_DRAWDOWN_FLOOR,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            dd_window=dd_window,
            max_drawdown_floor=max_drawdown_floor,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.dd_window = int(dd_window)
        self.max_drawdown_floor = float(max_drawdown_floor)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
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
            # Defensive: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = max(self.momentum_window, self.dd_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # Max-drawdown quality screen over last dd_window bars
                if len(col) >= self.dd_window:
                    dd_prices = col.values[-self.dd_window:]
                else:
                    dd_prices = col.values
                max_dd = _compute_max_drawdown(dd_prices)
                if max_dd > self.max_drawdown_floor:
                    continue  # reject stocks in active breakdown

                # 126d momentum
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
                # Not enough quality candidates — fall back to IEF
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


NAME = "sp500_maxdd_quality_momentum"
HYPOTHESIS = (
    "SP500 momentum with per-stock 21d max-drawdown quality screen: exclude SP500 stocks "
    "with max-drawdown > 12% over last 21 days (falling knives filter); rank remaining by "
    "126d return; hold top-15 above SPY 200d SMA; inverse-vol weighted; IEF defensive; "
    "biweekly rebalance — per-stock drawdown screen prevents momentum in breakdown stocks, "
    "distinct from RSI screen (gen9) and near-high filter (gen6)"
)

UNIVERSE = _universe

STRATEGY = SP500MaxDDQualityMomentum()
