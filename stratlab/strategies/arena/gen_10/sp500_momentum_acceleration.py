"""SP500 momentum with short-term acceleration filter — gen_10 sonnet-7

Hypothesis: Apply a 21d return acceleration filter before 126d momentum
ranking. Exclude stocks where the 21d return < 0 while the 126d return > 0
(momentum decelerating — the stock had great 6-month momentum but is now
rolling over). Hold top-15 of the remaining above their own 126d SMA.
Inverse-vol weighted. SPY 200d outer bear gate to TLT. Biweekly rebalance.

Rationale:
  - Classic momentum chases past winners. Decelerating names (positive
    6-month return but negative 1-month return) are often early signs of
    trend exhaustion or fundamental deterioration.
  - The acceleration filter is different from RSI (RSI<35 is deep oversold;
    21d<0 can happen at RSI 45-55 — early deceleration before RSI signals).
  - Using the stock's own 126d SMA as a trend gate (not just SPY 200d)
    adds a per-name intermediate trend filter.
  - This is a selection-quality mechanism: it prunes candidates from the
    momentum top-list based on their own recent dynamics, not market regime.
  - Different from gen9_sp500_rsi_quality_momentum (RSI<35 threshold) in
    that acceleration filter fires at 21d<0 even when RSI remains 45-60.

Design:
  - Warmup: max(126d momentum, 126d SMA, 21d acceleration) + buffer.
  - Acceleration filter: 21d return >= 0 (recent continuation confirmed).
  - Momentum filter: 126d return (intermediate trend strength).
  - Trend gate: stock above its own 126d SMA.
  - Market gate: SPY 200d SMA bear -> TLT defensive.
  - Inverse-vol sizing (21d realized vol).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # 6-month momentum
ACCEL_WINDOW = 21         # acceleration check window (1 month)
STOCK_TREND_WINDOW = 126  # per-stock SMA filter
VOL_WINDOW = 21           # inverse-vol lookback
SPY_TREND_WINDOW = 200    # outer bear gate
TOP_K = 15
EXPOSURE = 0.97


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "TLT"]


UNIVERSE = _universe


class SP500MomentumAcceleration(Strategy):
    """SP500 126d momentum with 21d acceleration filter; per-stock 126d SMA
    trend gate; inverse-vol weighted; SPY 200d outer bear gate to TLT;
    biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        accel_window: int = ACCEL_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            accel_window=accel_window,
            stock_trend_window=stock_trend_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.accel_window = int(accel_window)
        self.stock_trend_window = int(stock_trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.stock_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d outer bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Need window for momentum + acceleration + vol + trend
            need = max(self.momentum_window, self.stock_trend_window) + self.vol_window + 10
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in ("SPY", "TLT"):
                        continue
                    col = prices[sym].dropna()
                    n = len(col)
                    if n < self.momentum_window + self.vol_window + 2:
                        continue

                    # 126d momentum
                    p_end = float(col.iloc[-1])
                    p_126 = float(col.iloc[-self.momentum_window])
                    if p_126 <= 0 or not np.isfinite(p_126) or not np.isfinite(p_end):
                        continue
                    mom_ret = p_end / p_126 - 1.0
                    if not np.isfinite(mom_ret):
                        continue

                    # Acceleration filter: 21d return must be >= 0
                    # (if stock had big 6m return but is rolling over in 1m, skip it)
                    if n < self.accel_window + 2:
                        continue
                    p_21 = float(col.iloc[-self.accel_window])
                    if p_21 <= 0:
                        continue
                    accel_ret = p_end / p_21 - 1.0
                    # Exclude: positive 126d momentum BUT negative 21d (decelerating)
                    if mom_ret > 0 and accel_ret < 0:
                        continue
                    # Also exclude stocks with negative 126d momentum entirely
                    if mom_ret <= 0:
                        continue

                    # Per-stock 126d SMA trend gate
                    if n < self.stock_trend_window + 2:
                        continue
                    sma_126 = float(col.iloc[-self.stock_trend_window:].mean())
                    if p_end <= sma_126:
                        continue

                    # Inverse-vol sizing (21d realized vol)
                    tail = col.values[-(self.vol_window + 1):]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = mom_ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    # Not enough quality candidates
                    if "TLT" in live:
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


NAME = "sp500_momentum_acceleration"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock 21d return acceleration filter: exclude stocks where "
    "21d return < 0 but 126d return > 0 (momentum decelerating); hold top-15 remaining stocks "
    "above their 126d SMA; inverse-vol weighted; SPY 200d outer bear gate to TLT; biweekly "
    "rebalance — acceleration filter is orthogonal to RSI quality filter"
)

STRATEGY = SP500MomentumAcceleration()
