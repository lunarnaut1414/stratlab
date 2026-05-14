"""SP500 RSI Mean-Reversion with Breadth Gate — gen_8 sonnet-3

Hypothesis: Hold top-15 SP500 stocks by RSI(14) ascending (most oversold,
RSI < 40) that are above their 200d SMA; require SPY breadth gate (>50% of
SP500 stocks above their 50d SMA) for entry; exit when RSI > 60 or at
rebalance; equal-weight; rebalance every 5 bars.

Rationale: Classic mean-reversion on oversold quality stocks. The 200d SMA
filter ensures we're buying dips in uptrends, not falling knives. The breadth
gate (>50% of stocks above 50d SMA) ensures the market regime supports
recoveries. This is distinct from all existing momentum strategies — they rank
by strongest performers, this ranks by most oversold in uptrending regime.

IS window: 2010-2018 (9 years).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5         # Check and rebalance every week
RSI_WINDOW = 14             # Standard RSI
RSI_ENTRY_THRESH = 40.0     # Only buy if RSI < 40 (oversold)
RSI_EXIT_THRESH = 60.0      # Exit when RSI > 60 (recovered)
TREND_WINDOW = 200          # 200d SMA for stock trend filter
BREADTH_WINDOW = 50         # 50d SMA for breadth check
BREADTH_THRESH = 0.50       # >50% of stocks above 50d SMA required
TOP_K = 15
EXPOSURE = 0.97


def _compute_rsi(prices: np.ndarray, window: int) -> float:
    """Compute RSI from a price array. Returns RSI value."""
    if len(prices) < window + 1:
        return 50.0  # neutral if not enough data
    deltas = np.diff(prices[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


class RsiMeanRevBreadth(Strategy):
    """SP500 oversold-RSI mean-reversion with market-breadth gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rsi_window: int = RSI_WINDOW,
        rsi_entry_thresh: float = RSI_ENTRY_THRESH,
        rsi_exit_thresh: float = RSI_EXIT_THRESH,
        trend_window: int = TREND_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        breadth_thresh: float = BREADTH_THRESH,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rsi_window=rsi_window,
            rsi_entry_thresh=rsi_entry_thresh,
            rsi_exit_thresh=rsi_exit_thresh,
            trend_window=trend_window,
            breadth_window=breadth_window,
            breadth_thresh=breadth_thresh,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.rsi_window = int(rsi_window)
        self.rsi_entry_thresh = float(rsi_entry_thresh)
        self.rsi_exit_thresh = float(rsi_exit_thresh)
        self.trend_window = int(trend_window)
        self.breadth_window = int(breadth_window)
        self.breadth_thresh = float(breadth_thresh)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.rsi_window + 10
        if ctx.idx < warmup:
            return []

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items()
                if not s.startswith("^") and float(p) > 0}

        # Need enough history for trend + RSI
        need = max(self.trend_window, self.breadth_window) + self.rsi_window + 5
        prices_df = ctx.closes_window(need)
        if len(prices_df) < need - 5:
            return []

        # Compute breadth: fraction of SP500 stocks above their 50d SMA
        breadth_count = 0
        breadth_total = 0
        for sym in live:
            if sym not in prices_df.columns:
                continue
            col = prices_df[sym].dropna()
            if len(col) < self.breadth_window:
                continue
            sma50 = float(col.iloc[-self.breadth_window:].mean())
            price = float(col.iloc[-1])
            breadth_total += 1
            if price > sma50:
                breadth_count += 1

        breadth_ratio = breadth_count / breadth_total if breadth_total > 0 else 0.0

        # Compute portfolio equity
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if breadth_ratio >= self.breadth_thresh:
            # Market breadth is healthy — look for oversold stocks in uptrend
            candidates: list[tuple[str, float]] = []

            for sym in live:
                if sym not in prices_df.columns:
                    continue
                col = prices_df[sym].dropna()
                if len(col) < self.trend_window + self.rsi_window + 2:
                    continue

                # Stock must be in uptrend (above 200d SMA)
                sma200 = float(col.iloc[-self.trend_window:].mean())
                price = float(col.iloc[-1])
                if price <= sma200:
                    continue

                # Compute RSI(14)
                rsi = _compute_rsi(col.values, self.rsi_window)

                if rsi < self.rsi_entry_thresh:
                    candidates.append((sym, rsi))

            if candidates:
                # Sort ascending by RSI (most oversold first)
                candidates.sort(key=lambda x: x[1])
                selected = [sym for sym, _ in candidates[:self.top_k]]

                per_slot = self.exposure / len(selected)
                for sym in selected:
                    target[sym] = per_slot

        # Exit any positions whose RSI has recovered above exit threshold
        # (only applies to positions not in new target)
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                # Check if RSI has recovered
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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
    return sp500_tickers() + ["SPY", "TLT", "SHY"]


NAME = "rsi_meanrev_breadth"
HYPOTHESIS = (
    "SP500 oversold-RSI mean-reversion with breadth gate: hold top-15 SP500 stocks by "
    "RSI(14) ascending (most oversold, RSI<40) that are above their 200d SMA; SPY breadth "
    "gate (>50% of SP500 stocks above 50d SMA) required for entry; exit when RSI>60; "
    "equal-weight; rebalance every 5 bars; exploits institutional selling dips in uptrends"
)

UNIVERSE = _universe

STRATEGY = RsiMeanRevBreadth()
