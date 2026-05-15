"""SP500 momentum with RSI quality filter.

Hypothesis: Exclude SP500 stocks with RSI(14) < 35 before applying 126d
momentum ranking. This avoids "falling knife" momentum — stocks that rank
high on momentum because they were extremely high-momentum before a sharp
reversal, but are now technically broken. The RSI floor acts as a quality
screen: we only buy momentum in names that are still in upward-pressure
mode (not oversold capitulation).

Rationale:
  - Pure momentum strategies can chase names entering breakdown (high 6-month
    return but RSI < 35 = recent reversal): RSI exclusion avoids these.
  - Near-52w-high filter (gen6) helps but uses a 252d window — RSI 14d is
    faster and picks up recent shifts.
  - Combining 126d momentum (intermediate trend) with RSI > 35 quality screen
    is different from all leaderboard entries.

Design:
  - Compute RSI(14) for each SP500 stock.
  - Only rank stocks with RSI >= 35 (not in deep oversold territory).
  - Rank by 126d momentum; hold top-15 above SPY 200d SMA.
  - Inverse-vol weighting for position sizing.
  - Defensive: IEF when SPY below 200d SMA.
  - Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 126    # ~6 months
RSI_WINDOW = 14          # standard RSI lookback
RSI_FLOOR = 35.0         # exclude stocks with RSI below this
VOL_WINDOW = 21          # for inverse-vol weights
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97


def _compute_rsi(prices: "np.ndarray", window: int) -> float:
    """Compute RSI(window) from a 1d close-price array.

    Returns NaN if insufficient data.
    """
    if len(prices) < window + 1:
        return float("nan")
    deltas = np.diff(prices[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class SP500RsiQualityMomentum(Strategy):
    """SP500 126d momentum with RSI >= 35 quality screen; inverse-vol weighted;
    SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rsi_window: int = RSI_WINDOW,
        rsi_floor: float = RSI_FLOOR,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rsi_window=rsi_window,
            rsi_floor=rsi_floor,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rsi_window = int(rsi_window)
        self.rsi_floor = float(rsi_floor)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.rsi_window + 10
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
            # Need lookback for momentum + RSI
            need = self.momentum_window + self.rsi_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < need - 5:
                    continue

                # RSI quality filter
                rsi_val = _compute_rsi(col.values, self.rsi_window)
                if not np.isfinite(rsi_val) or rsi_val < self.rsi_floor:
                    continue

                # 126d momentum
                if len(col) < self.momentum_window + 2:
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


NAME = "sp500_rsi_quality_momentum"
HYPOTHESIS = (
    "SP500 momentum with RSI quality filter: exclude stocks with 14d RSI < 35 (avoid momentum "
    "in oversold / fundamentally troubled names); rank remaining by 126d momentum; hold top-15 "
    "above 200d SMA; inverse-vol weighted; SPY 200d outer trend gate; IEF defensive; biweekly "
    "rebalance — RSI exclusion prevents chasing falling knives inside momentum ranking"
)

UNIVERSE = _universe

STRATEGY = SP500RsiQualityMomentum()
