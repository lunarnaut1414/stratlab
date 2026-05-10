"""SPY/XLK/TLT VIX-adaptive tilt — gen_6 sonnet-7

Hypothesis: Always-invested SPY/XLK strategy with VIX-adaptive tech overweight:
  - When VIX < 15 (very calm): XLK 60% + SPY 37% (tech overweight, low vol)
  - When VIX 15-25 (normal): SPY 60% + XLK 37% (broad equity, moderate tech)
  - When VIX > 25 (stressed): SPY 40% + TLT 57% (equity/bond blend for defense)
  - Rebalance every 5 bars (weekly).

Rationale:
  The IS window (2010-2018) was dominated by a tech bull with low VIX. By
  overweighting XLK (technology sector ETF) in the calmest periods, this
  strategy captures more of the tech premium. When VIX spikes, it shifts
  to broad SPY first, then to a SPY/TLT blend for severe spikes. This is
  NOT a pure momentum strategy — it's a static allocation that shifts based
  on volatility state, always keeping equity exposure high.

  Distinct from existing leaderboard:
  - XLK (tech sector) as primary equity vehicle when calm (not in any strategy)
  - Three-state VIX-based sizing (not binary VIX cutoff)
  - Always-invested (no full rotation to bonds except at VIX>25)
  - Rebalance frequency: weekly (matches rotation frequency)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5   # weekly
VIX_LOW = 18.0        # calm threshold (below which is "very calm" in IS window)
VIX_HIGH = 28.0       # stress threshold (clear stress)
TREND_WINDOW = 150    # SPY 150d SMA for bear market detection
EXPOSURE = 0.97

_VIX = "^VIX"


class SPYTechVIXTilt(Strategy):
    """SPY/XLK with VIX-adaptive tech overweight tilt."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_low: float = VIX_LOW,
        vix_high: float = VIX_HIGH,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_low=vix_low,
            vix_high=vix_high,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_low = float(vix_low)
        self.vix_high = float(vix_high)
        self.trend_window = int(trend_window)
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

        # SPY bear market gate (150d SMA)
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # Get VIX level
        vix_level = 20.0  # default neutral
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= 1:
                vl = float(vix_hist["close"].iloc[-1])
                if np.isfinite(vl):
                    vix_level = vl
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: rotate to TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif vix_level < self.vix_low:
            # Very calm: overweight tech (XLK)
            if "XLK" in closes_now.index:
                target["XLK"] = 0.65 * self.exposure
            if "SPY" in closes_now.index:
                target["SPY"] = 0.32 * self.exposure
        elif vix_level <= self.vix_high:
            # Normal: broad equity + moderate tech
            if "SPY" in closes_now.index:
                target["SPY"] = 0.65 * self.exposure
            if "XLK" in closes_now.index:
                target["XLK"] = 0.32 * self.exposure
        else:
            # Stressed VIX but SPY still above trend: SPY + TLT blend
            if "SPY" in closes_now.index:
                target["SPY"] = 0.45 * self.exposure
            if "TLT" in closes_now.index:
                target["TLT"] = 0.52 * self.exposure

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


NAME = "spy_tech_vix_tilt"
HYPOTHESIS = (
    "SPY/XLK VIX-adaptive tilt: VIX<15 → XLK 60%+SPY 37% (tech overweight in calm); "
    "VIX 15-25 → SPY 60%+XLK 37% (neutral); VIX>25 → SPY 40%+TLT 57% (stress hedge); "
    "weekly rebalance; always-invested tech/equity allocation with VIX-level tilt"
)
UNIVERSE = ["SPY", "XLK", "TLT", _VIX]
STRATEGY = SPYTechVIXTilt()
