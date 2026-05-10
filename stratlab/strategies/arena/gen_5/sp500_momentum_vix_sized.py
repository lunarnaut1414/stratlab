"""SP500 Momentum with VIX-Based Position Sizing — gen_5 sonnet-5

Hypothesis: Rank SP500 stocks by 42-day momentum, hold top-15 above their
200d SMA. Scale total exposure based on inverse VIX level: fully invested
when VIX is low (calm), reduce to 50% when VIX is elevated (>25).
Bi-weekly rebalance.

Rationale: Cross-sectional momentum in SP500 is well-documented. Adding
VIX-based exposure scaling provides automatic deleveraging during high-stress
periods while keeping full exposure during trending low-vol environments.
The 42-day window (roughly 2 months) strikes a balance between noise and
persistence.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 42
TREND_WINDOW = 200
TOP_K = 15
REBALANCE_DAYS = 10   # Bi-weekly
VIX_LOW_THRESH = 15.0
VIX_HIGH_THRESH = 25.0
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["^VIX"]


UNIVERSE = _universe


class Sp500MomentumVixSized(Strategy):
    """Top-15 SP500 momentum stocks with VIX-adaptive position sizing."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOMENTUM_WINDOW, TREND_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        # Get VIX for exposure scaling
        vix_hist = ctx.history("^VIX")
        vix_val = 20.0  # default if unavailable
        if len(vix_hist) >= 2:
            vix_val = float(vix_hist["close"].iloc[-1])

        # Scale exposure inversely to VIX
        if vix_val <= VIX_LOW_THRESH:
            exposure = MAX_EXPOSURE
        elif vix_val >= VIX_HIGH_THRESH:
            exposure = MIN_EXPOSURE
        else:
            # Linear interpolation
            frac = (vix_val - VIX_LOW_THRESH) / (VIX_HIGH_THRESH - VIX_LOW_THRESH)
            exposure = MAX_EXPOSURE - frac * (MAX_EXPOSURE - MIN_EXPOSURE)

        closes = ctx.closes()
        if closes.empty:
            return []

        live = {s: float(closes[s]) for s in closes.index
                if closes[s] > 0 and not s.startswith("^")}

        prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
        if len(prices_window) < MOMENTUM_WINDOW:
            return []

        # Compute 42-day momentum for SP500 stocks
        scores: dict[str, float] = {}
        for sym in live:
            if sym not in prices_window.columns:
                continue
            col = prices_window[sym].dropna()
            if len(col) < MOMENTUM_WINDOW:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-MOMENTUM_WINDOW])
            if p_start <= 0:
                continue
            ret = p_end / p_start - 1.0
            if np.isfinite(ret):
                scores[sym] = ret

        if len(scores) < TOP_K:
            return []

        # Rank and take top-K with trend filter
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        selected = []
        for sym, _ in ranked:
            if len(selected) >= TOP_K:
                break
            hist = ctx.history(sym)
            if len(hist) < TREND_WINDOW:
                continue
            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
            price = live.get(sym, 0.0)
            if price > sma:
                selected.append(sym)

        if not selected:
            # Exit all and hold cash
            orders = []
            for sym, pos in list(ctx.positions.items()):
                if pos.size > 0:
                    orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))
            return orders

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        per_slot = equity * exposure / len(selected)
        target: dict[str, int] = {}
        for sym in selected:
            price = live.get(sym, 0.0)
            if price > 0:
                shares = int(per_slot / price)
                if shares > 0:
                    target[sym] = shares

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, tgt in target.items():
            current = ctx.position(sym).size
            delta = tgt - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sp500_momentum_vix_sized"
HYPOTHESIS = (
    "Top-15 SP500 stocks by 42-day momentum above 200d SMA, with VIX-adaptive exposure: "
    "97% invested when VIX<15, linearly scaled down to 50% when VIX>25. Bi-weekly rebalance."
)

STRATEGY = Sp500MomentumVixSized()
