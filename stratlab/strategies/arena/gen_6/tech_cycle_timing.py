"""Tech-cycle momentum timing strategy.

Hypothesis: Use XLK (tech) + XLY (consumer disc) composite 63-day return as
a growth-cycle regime signal, then allocate to QQQ/SPY/TLT accordingly.

Regime logic:
- Combined score > 15%: growth cycle expanding -> QQQ 95%
- Score 0-15%: moderate growth -> SPY 75% + TLT 22%
- Score < 0%: contraction -> TLT 60% + SHY 35%

Secondary gate: SPY 200d SMA -- if SPY below SMA, override to contraction
regardless of sector momentum score.

Rebalance every 5 bars (weekly).

Structural distinctions:
- Uses sector ETF momentum (XLK/XLY composite) as regime signal
- Routes exposure through broad index ETFs (QQQ/SPY/TLT), not stocks
- Different signal than VIX-level or credit-spread regime
- Tech+consumer disc cycle proxy captures leading economic indicators
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SIGNAL_ETFS = ["XLK", "XLY"]
MOMENTUM_WINDOW = 63
HIGH_THRESHOLD = 0.15   # >15% combined score = growth expansion
LOW_THRESHOLD = 0.0     # <0% = contraction
TREND_WINDOW = 200
REBALANCE_EVERY = 5
EXPOSURE = 0.97

# Allocations per regime
EXPANSION = {"QQQ": 0.97}
MODERATE = {"SPY": 0.97}       # no TLT drag in moderate regime
CONTRACTION = {"TLT": 0.60, "SHY": 0.35}


class TechCycleTiming(Strategy):
    """Tech-cycle ETF timing: XLK+XLY composite signal -> QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
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

        # SPY trend gate
        bull = True
        try:
            spy_hist = ctx.history("SPY")
            spy_c = spy_hist["close"].dropna()
            if len(spy_c) >= self.trend_window:
                bull = float(spy_c.iloc[-1]) > float(spy_c.iloc[-self.trend_window:].mean())
        except Exception:
            pass

        # Compute composite tech+consumer momentum
        score_sum = 0.0
        score_count = 0
        for sym in SIGNAL_ETFS:
            try:
                hist = ctx.history(sym)
                c = hist["close"].dropna()
                if len(c) >= self.momentum_window:
                    ret = float(c.iloc[-1] / c.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        score_sum += ret
                        score_count += 1
            except Exception:
                pass

        # Determine regime
        raw_alloc: dict[str, float]
        if not bull:
            raw_alloc = dict(CONTRACTION)
        elif score_count == 0:
            raw_alloc = dict(MODERATE)
        else:
            avg_score = score_sum / score_count
            if avg_score > self.high_threshold:
                raw_alloc = dict(EXPANSION)
            elif avg_score < self.low_threshold:
                raw_alloc = dict(CONTRACTION)
            else:
                raw_alloc = dict(MODERATE)

        # Scale to exposure
        target = {sym: wt * self.exposure for sym, wt in raw_alloc.items() if sym in live}

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


UNIVERSE = ["SPY", "QQQ", "TLT", "SHY", "XLK", "XLY"]
NAME = "tech_cycle_timing"
HYPOTHESIS = (
    "Tech-cycle QQQ/SPY timing v2: XLK+XLY 63d composite score; "
    "score>15% QQQ 95%; score 0-15% SPY 95%; score<0 or SPY<200d SMA TLT 60%+SHY 35%; "
    "no bond drag in moderate regime; weekly rebalance"
)
STRATEGY = TechCycleTiming()
