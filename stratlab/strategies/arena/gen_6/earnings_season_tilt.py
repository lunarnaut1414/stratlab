"""Earnings season tilt strategy.

Hypothesis: Hold QQQ (high-beta tech) during earnings months (January,
April, July, October) when VIX is calm (<22); hold SPY otherwise.
Rotate to TLT+SHY when SPY is in bear market (below 200d SMA).

Rationale:
- Major US earnings seasons occur in Jan, Apr, Jul, Oct (the "reporting months")
- Markets tend to exhibit pre-earnings drift and earnings-beat-driven lift
- Tech/growth stocks (QQQ) benefit more from positive earnings surprises
- VIX gate ensures we only tilt up during calm earnings seasons

Structural distinctions vs existing leaderboard:
- Calendar signal based on earnings seasons (different from halloween/TOM)
- QQQ in earnings months vs SPY in non-earnings months (not VIX regime purely)
- Low trade count (monthly switches) -- different liquidity profile
- Does NOT pick individual stocks; pure ETF rotation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

EARNINGS_MONTHS = {1, 4, 7, 10}   # Jan, Apr, Jul, Oct
VIX_THRESHOLD = 22.0
TREND_WINDOW = 200
REBALANCE_EVERY = 5  # weekly rebalance to catch month transitions
EXPOSURE = 0.97

# Allocations
EARNINGS_ALLOC = {"QQQ": 1.0}
NORMAL_ALLOC = {"SPY": 1.0}
BEAR_ALLOC = {"TLT": 0.60, "SHY": 0.37}


class EarningsSeasonTilt(Strategy):
    """QQQ/SPY earnings-season tilt with VIX gate and SPY 200d bear protection."""

    def __init__(
        self,
        vix_threshold: float = VIX_THRESHOLD,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            vix_threshold=vix_threshold,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.vix_threshold = float(vix_threshold)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 5
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

        # VIX level
        vix_level = float("nan")
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        # Current month
        current_month = ctx.timestamp.month

        # Determine allocation
        raw_alloc: dict[str, float]
        if not bull:
            raw_alloc = dict(BEAR_ALLOC)
        elif (current_month in EARNINGS_MONTHS
              and (np.isnan(vix_level) or vix_level < self.vix_threshold)):
            # Earnings season + calm VIX: tilt to QQQ
            raw_alloc = dict(EARNINGS_ALLOC)
        else:
            # Non-earnings month or VIX elevated: stay in SPY
            raw_alloc = dict(NORMAL_ALLOC)

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


UNIVERSE = ["SPY", "QQQ", "TLT", "SHY", "^VIX"]
NAME = "earnings_season_tilt"
HYPOTHESIS = (
    "Earnings season tilt: hold QQQ 95% during earnings months (Jan/Apr/Jul/Oct) "
    "when VIX<22; hold SPY 95% otherwise; TLT 60%+SHY 35% when SPY below 200d SMA; "
    "captures earnings-season beta lift; weekly rebalance"
)
STRATEGY = EarningsSeasonTilt()
