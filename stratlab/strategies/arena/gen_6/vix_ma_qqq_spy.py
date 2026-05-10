"""VIX 5d/20d MA crossover → QQQ/SPY/TLT — gen_6 sonnet-7

Hypothesis: Use VIX 5d MA vs 20d MA crossover as a short-term fear signal:
  - VIX 5d MA < VIX 20d MA (VIX declining, fear receding) AND SPY > 150d SMA:
    hold QQQ 97% (risk-on, declining fear = growth premium)
  - VIX 5d MA > VIX 20d MA (fear rising) but SPY still above 150d SMA:
    hold SPY 97% (less aggressive but still bullish)
  - SPY < 150d SMA (bear market): hold TLT 97%
  Rebalance every 3 bars.

Rationale:
  A falling VIX (5d MA < 20d MA) indicates fear is subsiding, which
  historically correlates with momentum continuation in growth/tech stocks.
  When fear starts rising (VIX increasing vs its own MA), reducing from
  QQQ to SPY provides defensive repositioning while staying in equities.
  The SPY trend gate catches structural bear markets.

  Distinct from existing strategies:
  - VIX 5d vs 20d MA crossover (not absolute level like VIX<20 or VIX<25)
  - Declining VIX as QQQ signal (not rising VIX as sell signal)
  - 3-bar rebalance (more responsive than weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 3    # every 3 bars
VIX_FAST = 5           # VIX 5d MA
VIX_SLOW = 20          # VIX 20d MA
TREND_WINDOW = 150     # SPY 150d SMA
EXPOSURE = 0.97

_VIX = "^VIX"


class VIXMAQQQSpy(Strategy):
    """VIX 5d/20d MA crossover: declining VIX → QQQ, rising VIX → SPY, bear → TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_fast: int = VIX_FAST,
        vix_slow: int = VIX_SLOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_fast=vix_fast,
            vix_slow=vix_slow,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_fast = int(vix_fast)
        self.vix_slow = int(vix_slow)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vix_slow, self.trend_window) + 5
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
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # VIX MA crossover
        vix_declining = True  # assume declining if no data
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= self.vix_slow + 2:
                vix_close = vix_hist["close"].dropna().values
                vix_fast_ma = float(np.mean(vix_close[-self.vix_fast:]))
                vix_slow_ma = float(np.mean(vix_close[-self.vix_slow:]))
                vix_declining = vix_fast_ma < vix_slow_ma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif vix_declining:
            # VIX declining (fear receding): QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # VIX rising but SPY still in trend: SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

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


NAME = "vix_ma_qqq_spy"
HYPOTHESIS = (
    "VIX 5d/20d MA crossover → QQQ/SPY/TLT: declining VIX (5d<20d MA) AND SPY>150d → QQQ 97%; "
    "rising VIX but SPY bull → SPY 97%; SPY<150d SMA → TLT 97%; 3-bar rebalance; "
    "VIX direction (not level) as QQQ vs SPY selector"
)
UNIVERSE = ["QQQ", "SPY", "TLT", _VIX]
STRATEGY = VIXMAQQQSpy()
