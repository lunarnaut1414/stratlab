"""SPY momentum signal → QQQ/SPY/TLT allocation — gen_6 sonnet-7

Hypothesis: Use SPY 63d return as a momentum signal to select equity vehicle:
  - SPY 63d return > 8% AND VIX < 20: strong bull → hold QQQ 97% (max tech exposure)
  - SPY 63d return > 0% AND VIX < 25: moderate bull → hold SPY 97% (broad market)
  - SPY 63d return < 0% OR VIX > 25: weak / bearish → hold TLT 97%
  Rebalance every 5 bars.

Rationale:
  The strength of recent market momentum (not just direction) is a better
  indicator of whether to hold high-beta tech (QQQ) or broad market (SPY).
  When momentum is strong (>8%) and fear is low (VIX<20), QQQ's higher beta
  generates superior returns. When momentum is moderate (0-8%), broad SPY
  captures the trend with less sector concentration. This creates a natural
  tiered exposure system.

  Distinct from existing leaderboard strategies:
  - Uses SPY 63d return as a MAGNITUDE signal (not just cross-over)
  - Different from JNK-gated QQQ (uses equity momentum, not credit signal)
  - Different from breadth-timing (uses SPY return, not % of stocks above MA)
  - QQQ only in strong-bull regime (not always)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
MOMENTUM_WINDOW = 63     # SPY 63d return
STRONG_BULL_THRESH = 0.08  # >8% = strong bull
VIX_CALM = 20.0          # VIX threshold for QQQ
VIX_MODERATE = 25.0      # VIX threshold for SPY
EXPOSURE = 0.97

_VIX = "^VIX"


class SPYMomentumVIXQQQ(Strategy):
    """SPY 63d momentum magnitude + VIX level → QQQ/SPY/TLT tiered allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        strong_bull_thresh: float = STRONG_BULL_THRESH,
        vix_calm: float = VIX_CALM,
        vix_moderate: float = VIX_MODERATE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            strong_bull_thresh=strong_bull_thresh,
            vix_calm=vix_calm,
            vix_moderate=vix_moderate,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.strong_bull_thresh = float(strong_bull_thresh)
        self.vix_calm = float(vix_calm)
        self.vix_moderate = float(vix_moderate)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 5
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

        # SPY 63d momentum
        spy_mom = 0.0
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.momentum_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_mom = float(spy_close[-1] / spy_close[-self.momentum_window] - 1.0)
        except Exception:
            pass

        # VIX level
        vix_level = 20.0
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= 1:
                vl = float(vix_hist["close"].iloc[-1])
                if np.isfinite(vl):
                    vix_level = vl
        except Exception:
            pass

        target: dict[str, float] = {}

        if spy_mom >= self.strong_bull_thresh and vix_level < self.vix_calm:
            # Strong bull: QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        elif spy_mom > 0 and vix_level < self.vix_moderate:
            # Moderate bull: SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            # Weak/bearish or elevated VIX: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure

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


NAME = "spy_momentum_vix_qqq"
HYPOTHESIS = (
    "SPY 63d momentum magnitude → QQQ/SPY/TLT: SPY return >8% AND VIX<20 → QQQ 97%; "
    "SPY return >0% AND VIX<25 → SPY 97%; SPY down OR VIX>25 → TLT 97%; "
    "weekly rebalance; SPY return magnitude (not just direction) as equity selection signal"
)
UNIVERSE = ["QQQ", "SPY", "TLT", _VIX]
STRATEGY = SPYMomentumVIXQQQ()
