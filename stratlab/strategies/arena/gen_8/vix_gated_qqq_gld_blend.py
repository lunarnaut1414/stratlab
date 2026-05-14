"""VIX-Level Gated QQQ/GLD Blend — gen_8 sonnet-2

Hypothesis: Use VIX absolute level (not direction) with a GLD blend to create
a smooth equity-gold allocation. At low VIX, overweight QQQ and hold a small
GLD position. At elevated VIX, reduce QQQ and increase GLD+TLT defensively.

Unlike pure equity VIX-gated strategies, this always holds some GLD, creating a
more distinct return profile (lower correlation to SP500 equity strategies).

Regime tiers (using fixed absolute VIX levels):
  - VIX < 16 (very calm): QQQ 87% + GLD 10% (growth tilt with inflation hedge)
  - VIX 16-22 (normal): QQQ 70% + GLD 20% + IEF 7% (balanced)
  - VIX 22-30 (elevated): SPY 50% + TLT 30% + GLD 17% (cautious)
  - VIX > 30 (stressed): TLT 60% + GLD 37% (defensive)
  - SPY 200d SMA bear: TLT 60% + GLD 37% override

Rationale:
- GLD hedge reduces correlation to pure equity momentum strategies
- Always-holding some GLD in calm regimes provides inflation protection
- Distinct signal (VIX absolute level + GLD blend) vs JNK/credit strategies
- More differentiated return profile than binary equity/bond switchers

Rebalance: every 5 bars (weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
TREND_WINDOW = 200        # SPY 200d SMA outer gate
VIX_CALM = 16.0           # below this: very calm
VIX_NORMAL = 22.0         # below this: normal; above is elevated
VIX_STRESSED = 30.0       # above this: stressed
EXPOSURE = 0.97
_VIX = "^VIX"
_QQQ = "QQQ"
_SPY = "SPY"
_TLT = "TLT"
_GLD = "GLD"
_IEF = "IEF"


class VIXGatedQQQGLDBlend(Strategy):
    """VIX-level gated QQQ+GLD blend: 4 tiers from calm to stressed."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vix_calm: float = VIX_CALM,
        vix_normal: float = VIX_NORMAL,
        vix_stressed: float = VIX_STRESSED,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vix_calm=vix_calm,
            vix_normal=vix_normal,
            vix_stressed=vix_stressed,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vix_calm = float(vix_calm)
        self.vix_normal = float(vix_normal)
        self.vix_stressed = float(vix_stressed)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        # Get VIX level
        vix_level = 20.0  # default to normal
        try:
            vix_hist = ctx.history(_VIX)
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= 2:
                vix_level = float(vix_close.iloc[-1])
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target weights
        raw_target: dict[str, float] = {}

        if not bull or vix_level > self.vix_stressed:
            # Bear market or VIX very stressed: TLT + GLD defensive
            raw_target[_TLT] = 0.62
            raw_target[_GLD] = 0.38
        elif vix_level > self.vix_normal:
            # Elevated VIX: cautious blend
            raw_target[_SPY] = 0.52
            raw_target[_TLT] = 0.30
            raw_target[_GLD] = 0.18
        elif vix_level > self.vix_calm:
            # Normal VIX: balanced QQQ+GLD+IEF
            raw_target[_QQQ] = 0.72
            raw_target[_GLD] = 0.21
            raw_target[_IEF] = 0.07
        else:
            # Very calm: mostly QQQ with GLD hedge
            raw_target[_QQQ] = 0.89
            raw_target[_GLD] = 0.11

        # Apply exposure cap and filter to live symbols
        total_w = sum(raw_target.values())
        target = {
            s: (w / total_w) * self.exposure
            for s, w in raw_target.items()
            if s in live
        }

        orders: list[Order] = []

        # Exit positions not in target
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


UNIVERSE = [_QQQ, _SPY, _TLT, _GLD, _IEF, _VIX]

NAME = "vix_gated_qqq_gld_blend"
HYPOTHESIS = (
    "VIX-level gated QQQ+GLD blend: 4-tier VIX allocation; calm (<16) QQQ 89%+GLD 11%; "
    "normal (16-22) QQQ 72%+GLD 21%+IEF 7%; elevated (22-30) SPY 52%+TLT 30%+GLD 18%; "
    "stressed (>30 or SPY<200d) TLT 62%+GLD 38%; GLD hedge throughout reduces corr to equity cluster"
)

STRATEGY = VIXGatedQQQGLDBlend()
