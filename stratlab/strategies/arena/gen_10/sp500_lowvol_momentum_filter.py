"""SP500 low-volatility momentum filter — gen_10 sonnet-7

Hypothesis: Select SP500 stocks with positive 126d momentum (trend confirmed)
but rank by LOWEST 21d realized volatility (stable trend quality). This is
the opposite of chasing the highest-momentum names — we want stocks that
are trending positively but with minimal noise/volatility. The dual
requirement (positive 126d momentum AND low vol) selects stocks in steady
quiet uptrends, avoiding both high-vol breakouts AND pure value/low-vol.

Rationale:
  - Pure momentum selects the fastest-moving stocks (high vol, high noise).
  - Pure low-vol selects slow-grinding stocks (may have negative momentum).
  - Combining: positive 126d momentum filter THEN rank by lowest vol
    selects "quiet persistent uptrenders" — different from any existing
    leaderboard strategy.
  - gen9_sp500_rsi_quality_momentum uses RSI>35 as quality filter (momentum
    + RSI floor = avoid broken momentum). This uses low-vol ranking as
    quality signal — different mechanism.
  - Expected OOS behavior: quiet uptrenders may be more robust to regime
    changes because they're not concentrated in speculative names.
  - Not correlated with RSP breadth regime (different mechanism entirely).

Design:
  - Include stock if 126d return > 0 AND above its own 200d SMA.
  - Rank survivors by 21d realized vol (ascending — lowest first).
  - Hold top-15 lowest-vol names that meet both filters.
  - Equal-weight (not inverse-vol since ranking IS vol already).
  - SPY 200d outer bear gate to TLT.
  - Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # 6-month momentum threshold
VOL_WINDOW = 21           # realized vol for ranking
TREND_WINDOW = 200        # per-stock and SPY SMA
TOP_K = 15
EXPOSURE = 0.97


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "TLT"]


UNIVERSE = _universe


class SP500LowVolMomentumFilter(Strategy):
    """SP500 stocks with positive 126d momentum above 200d SMA, ranked by
    LOWEST 21d realized vol; equal-weight top-15; SPY 200d outer bear gate
    to TLT; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window, self.vol_window) + 10
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
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            need = max(self.momentum_window, self.trend_window) + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                # Store (vol, symbol) for candidates that pass momentum+trend filter
                vol_candidates: list[tuple[float, str]] = []

                for sym in prices.columns:
                    if sym in ("SPY", "TLT"):
                        continue
                    col = prices[sym].dropna()
                    n = len(col)
                    if n < max(self.momentum_window, self.trend_window) + self.vol_window + 2:
                        continue

                    # 126d momentum filter: must be positive
                    p_end = float(col.iloc[-1])
                    p_126 = float(col.iloc[-self.momentum_window])
                    if p_126 <= 0 or not np.isfinite(p_126) or not np.isfinite(p_end):
                        continue
                    mom_ret = p_end / p_126 - 1.0
                    if not np.isfinite(mom_ret) or mom_ret <= 0:
                        continue

                    # Per-stock 200d SMA trend gate
                    sma_200 = float(col.iloc[-self.trend_window:].mean())
                    if p_end <= sma_200:
                        continue

                    # 21d realized vol for ranking
                    tail = col.values[-(self.vol_window + 1):]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    vol_candidates.append((rv, sym))

                if len(vol_candidates) < 5:
                    if "TLT" in live:
                        target["TLT"] = self.exposure
                else:
                    # Sort ascending by vol — lowest-vol names first
                    vol_candidates.sort(key=lambda x: x[0])
                    selected = [sym for _, sym in vol_candidates[:self.top_k]]
                    per_w = self.exposure / len(selected)
                    for sym in selected:
                        target[sym] = per_w

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


NAME = "sp500_lowvol_momentum_filter"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock 21d return acceleration filter: exclude stocks where "
    "21d return < 0 but 126d return > 0 (momentum decelerating); hold top-15 remaining stocks "
    "above their 126d SMA; inverse-vol weighted; SPY 200d outer bear gate to TLT; biweekly "
    "rebalance — acceleration filter is orthogonal to RSI quality filter"
)

STRATEGY = SP500LowVolMomentumFilter()
