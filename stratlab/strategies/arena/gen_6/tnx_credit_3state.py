"""TNX-credit composite 3-state regime strategy — gen_6 sonnet-7

Hypothesis: Combine 10-year Treasury yield (^TNX) trend direction with JNK
credit health to create a 3-state macro regime:
  - State 1 (Risk-On): TNX 20d MA > 60d MA (rising rates = growth) AND JNK
    20d MA > 60d MA (credit healthy) → hold QQQ 97% (growth/cyclical tilt)
  - State 2 (Neutral/Defensive): Either signal negative → hold SPY 97%
  - State 3 (Risk-Off): Both signals bearish OR SPY below 100d SMA →
    hold TLT 60% + GLD 37% (duration + gold)
  Rebalance weekly.

Rationale:
  TNX direction and credit spread movement are economically related but
  operationally different signals. Rising rates + tightening credit = classic
  late-cycle expansion where growth stocks (QQQ) outperform. Falling rates +
  widening credit = risk-off where TLT+GLD provides protection. Single-signal
  strategies miss the joint state information.

  Distinct from all existing leaderboard strategies:
  - Combines TNX direction AND credit direction (not either alone)
  - Routes to QQQ (not SPY) in risk-on (captures higher growth premium)
  - TLT+GLD blend in risk-off (not just TLT or just bonds)
  - Uses SPY 100d SMA as override for bear market protection
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
FAST_MA = 20
SLOW_MA = 60
TREND_WINDOW = 100      # SPY 100d SMA as bear gate
EXPOSURE = 0.97

_TNX = "^TNX"


class TNXCreditComposite(Strategy):
    """3-state macro regime combining TNX trend + JNK credit for QQQ/SPY/TLT+GLD routing."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + self.trend_window + 5
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

        # --- TNX rate direction signal ---
        tnx_rising = False
        try:
            tnx_hist = ctx.history(_TNX)
            if len(tnx_hist) >= self.slow_ma + 2:
                tnx_close = tnx_hist["close"].dropna()
                tnx_fast = float(tnx_close.iloc[-self.fast_ma:].mean())
                tnx_slow = float(tnx_close.iloc[-self.slow_ma:].mean())
                tnx_rising = tnx_fast > tnx_slow
        except Exception:
            pass

        # --- JNK credit signal ---
        jnk_healthy = False
        try:
            jnk_hist = ctx.history("JNK")
            if len(jnk_hist) >= self.slow_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                jnk_fast = float(jnk_close.iloc[-self.fast_ma:].mean())
                jnk_slow = float(jnk_close.iloc[-self.slow_ma:].mean())
                jnk_healthy = jnk_fast > jnk_slow
        except Exception:
            pass

        # --- SPY bear market gate ---
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # --- 3-state regime ---
        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market override: TLT 60% + GLD 37%
            if "TLT" in closes_now.index:
                target["TLT"] = 0.60 * self.exposure
            if "GLD" in closes_now.index:
                target["GLD"] = 0.37 * self.exposure
        elif tnx_rising and jnk_healthy:
            # State 1: Risk-On - QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        elif not tnx_rising and not jnk_healthy:
            # State 3: Risk-Off - TLT + GLD
            if "TLT" in closes_now.index:
                target["TLT"] = 0.60 * self.exposure
            if "GLD" in closes_now.index:
                target["GLD"] = 0.37 * self.exposure
        else:
            # State 2: Neutral - SPY
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


NAME = "tnx_credit_3state"
HYPOTHESIS = (
    "TNX-credit composite 3-state regime: rising TNX 20d/60d MA AND JNK healthy → QQQ 97%; "
    "both bearish OR SPY<100d SMA → TLT 60%+GLD 37%; neutral (one signal) → SPY 97%; "
    "weekly rebalance; combines rate trend + credit health for QQQ/SPY/TLT+GLD routing"
)
UNIVERSE = ["QQQ", "SPY", "TLT", "GLD", "JNK", _TNX]
STRATEGY = TNXCreditComposite()
