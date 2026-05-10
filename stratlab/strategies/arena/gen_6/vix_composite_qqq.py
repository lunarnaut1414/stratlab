"""VIX composite signal → QQQ/SPY/TLT — gen_6 sonnet-7

Hypothesis: Combine two VIX signals for robust equity allocation:
  1. VIX LEVEL: VIX < 20 (calm absolute level)
  2. VIX TREND: VIX 10d MA < VIX 50d MA (fear has been falling recently)
  Both signals must be true to hold QQQ. If only one is true: hold SPY.
  If neither: hold SPY but reduce to 70% (hedge with TLT 27%). If SPY <
  150d SMA: full TLT. Rebalance weekly.

Rationale:
  Requiring BOTH a calm VIX level AND a declining VIX trend before going
  to QQQ is more robust than single-signal approaches. The 10d/50d MA
  crossover captures whether VIX has been trending down (not just momentarily
  low). Combined with the 200d SPY SMA for bear market protection.

  Distinct from existing:
  - Dual VIX signal (level AND trend) vs single-signal strategies
  - Uses 10d/50d VIX MA (not 5d/20d or absolute thresholds only)
  - Partial hedge in "neither VIX condition" (not full binary rotation)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5    # weekly
VIX_LEVEL_THRESH = 20.0    # absolute VIX calm threshold
VIX_FAST_MA = 10           # VIX 10d MA
VIX_SLOW_MA = 50           # VIX 50d MA
TREND_WINDOW = 150         # SPY 150d SMA
EXPOSURE = 0.97

_VIX = "^VIX"


class VIXCompositeQQQ(Strategy):
    """Dual VIX signal: level + trend for QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_level_thresh: float = VIX_LEVEL_THRESH,
        vix_fast_ma: int = VIX_FAST_MA,
        vix_slow_ma: int = VIX_SLOW_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_level_thresh=vix_level_thresh,
            vix_fast_ma=vix_fast_ma,
            vix_slow_ma=vix_slow_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_level_thresh = float(vix_level_thresh)
        self.vix_fast_ma = int(vix_fast_ma)
        self.vix_slow_ma = int(vix_slow_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vix_slow_ma, self.trend_window) + 5
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

        # VIX signals
        vix_level_calm = True
        vix_trend_calm = True
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= self.vix_slow_ma + 2:
                vix_close = vix_hist["close"].dropna().values
                current_vix = float(vix_close[-1])
                vix_level_calm = current_vix < self.vix_level_thresh
                vix_fast = float(np.mean(vix_close[-self.vix_fast_ma:]))
                vix_slow = float(np.mean(vix_close[-self.vix_slow_ma:]))
                vix_trend_calm = vix_fast < vix_slow
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif vix_level_calm and vix_trend_calm:
            # Both VIX conditions met: QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        elif vix_level_calm or vix_trend_calm:
            # One VIX condition: SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            # Neither VIX condition: SPY 70% + TLT 27%
            if "SPY" in closes_now.index:
                target["SPY"] = 0.70 * self.exposure
            if "TLT" in closes_now.index:
                target["TLT"] = 0.27 * self.exposure

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


NAME = "vix_composite_qqq"
HYPOTHESIS = (
    "VIX composite dual-signal: VIX<20 (level calm) AND VIX 10d<50d MA (trend declining) → QQQ 97%; "
    "one condition → SPY 97%; neither → SPY 70%+TLT 27%; SPY<150d → TLT 97%; "
    "weekly rebalance; dual VIX level+trend gate more robust than single-signal"
)
UNIVERSE = ["QQQ", "SPY", "TLT", _VIX]
STRATEGY = VIXCompositeQQQ()
