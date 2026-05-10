"""Multi-asset trend-following with ATR-based position sizing.

Hypothesis: Hold SPY/TLT/GLD/DBC each when price > 100d SMA; size each
position inversely by 20d ATR normalized by price (so each contributes equal
volatility risk). Cash in SHY for unused slots. Rebalance every 5 bars
(weekly), with 5% drift threshold to reduce churn.

Structural distinctions vs existing leaderboard:
- 4-asset multi-asset trend (not single-asset SPY or equity-only)
- ATR-normalized risk parity sizing (not equal-weight or inverse-vol)
- SMA-only gate per asset (no cross-asset VIX or credit signal)
- Diversified across equities, bonds, gold, commodities
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ASSETS = ["SPY", "TLT", "GLD", "DBC"]
CASH_PROXY = "SHY"
TREND_WINDOW = 100
ATR_WINDOW = 20
REBALANCE_EVERY = 5
DRIFT_THRESHOLD = 0.05
EXPOSURE = 0.95


class MultiAssetATRTrend(Strategy):
    """Multi-asset trend-following with ATR-normalized equal-risk sizing."""

    def __init__(
        self,
        trend_window: int = TREND_WINDOW,
        atr_window: int = ATR_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        drift_threshold: float = DRIFT_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            atr_window=atr_window,
            rebalance_every=rebalance_every,
            drift_threshold=drift_threshold,
            exposure=exposure,
        )
        self.trend_window = int(trend_window)
        self.atr_window = int(atr_window)
        self.rebalance_every = int(rebalance_every)
        self.drift_threshold = float(drift_threshold)
        self.exposure = float(exposure)
        self._last_weights: dict[str, float] = {}

    def _compute_atr(self, hist: pd.DataFrame, window: int) -> float:
        """True range ATR over the last `window` bars."""
        if len(hist) < window + 1:
            return float("nan")
        h = hist["high"].values
        l = hist["low"].values
        c = hist["close"].values
        tr = []
        for i in range(-window, 0):
            hi = float(h[i]) if not np.isnan(h[i]) else float(c[i])
            lo = float(l[i]) if not np.isnan(l[i]) else float(c[i])
            prev_c = float(c[i - 1]) if not np.isnan(c[i - 1]) else float(c[i])
            tr.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
        return float(np.mean(tr))

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.atr_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine which assets are in trend regime
        in_trend: dict[str, float] = {}  # sym -> inverse_atr_weight
        for sym in ASSETS:
            try:
                hist = ctx.history(sym)
            except Exception:
                continue
            if len(hist) < self.trend_window + 5:
                continue
            closes = hist["close"].dropna()
            if len(closes) < self.trend_window:
                continue
            current_price = float(closes.iloc[-1])
            sma = float(closes.iloc[-self.trend_window:].mean())
            if current_price <= sma:
                continue  # not in trend
            atr = self._compute_atr(hist, self.atr_window)
            if not np.isfinite(atr) or atr <= 0 or current_price <= 0:
                continue
            atr_normalized = atr / current_price  # fractional ATR
            in_trend[sym] = atr_normalized

        # Build target weights: inverse-ATR normalized weighting
        target: dict[str, float] = {}
        if in_trend:
            # Each asset gets weight proportional to 1/ATR_normalized
            inv_atr = {sym: 1.0 / v for sym, v in in_trend.items()}
            total_inv = sum(inv_atr.values())
            if total_inv > 0:
                for sym, inv_v in inv_atr.items():
                    target[sym] = self.exposure * inv_v / total_inv

        # Cash proxy for any unexposed capital
        cash_proxy_weight = self.exposure - sum(target.values())
        if CASH_PROXY in closes_now.index and cash_proxy_weight > 0.01:
            target[CASH_PROXY] = cash_proxy_weight

        # Check if rebalance needed based on drift
        needs_rebalance = False
        for sym, wt in target.items():
            old = self._last_weights.get(sym, 0.0)
            if abs(wt - old) > self.drift_threshold:
                needs_rebalance = True
                break
        for sym in self._last_weights:
            if sym not in target:
                needs_rebalance = True
                break

        if not needs_rebalance and self._last_weights:
            return []

        self._last_weights = dict(target)

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
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


NAME = "multiasset_atr_trend"
HYPOTHESIS = (
    "Multi-asset trend-following: hold SPY/TLT/GLD/DBC each when price > 100d SMA, "
    "sized inversely by 20d ATR normalized by price; cash in SHY for unused slots; "
    "weekly rebalance with 5% drift threshold"
)
UNIVERSE = ["SPY", "TLT", "GLD", "DBC", "SHY"]
STRATEGY = MultiAssetATRTrend()
