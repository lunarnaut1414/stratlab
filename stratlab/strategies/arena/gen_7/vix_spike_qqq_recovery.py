"""VIX Spike Recovery + SPY Trend Blend — gen_7 sonnet-7

Hypothesis: A VIX-regime blending strategy that distinguishes between
three states:
1. VIX > 30 (fear/crisis): wait 3 bars then aggressively buy QQQ (recovery play)
2. VIX 18-30 (normal): hold SPY 97%
3. VIX < 18 (calm/complacency): hold QQQ 97%

This exploits the well-documented mean-reversion of VIX after spikes above 30:
markets tend to rally 2-4 weeks after fear peaks. The 3-bar wait avoids buying
the initial crash, targeting the rebound instead.

Distinct from all existing strategies: no credit signal, no bond/momentum mix.
Uses VIX *level* in a 3-tier way where the highest tier is the most equity-aggressive
(counter-intuitive but based on crisis-recovery alpha).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "^VIX"]

REBALANCE_EVERY = 5        # weekly
VIX_CRISIS = 30.0          # VIX above this = fear peak = recovery entry
VIX_CALM = 18.0            # VIX below this = complacency = QQQ
SPIKE_LOOKBACK = 3         # Wait N bars after first VIX spike to enter
SPIKE_HOLD = 15            # Hold recovery position for this many bars
EXPOSURE = 0.97
TREND_WINDOW = 200
_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_VIX = "^VIX"


class VixSpikeQQQRecovery(Strategy):
    """3-tier VIX regime: crisis (recovery QQQ), normal (SPY), calm (QQQ)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_crisis: float = VIX_CRISIS,
        vix_calm: float = VIX_CALM,
        spike_hold: int = SPIKE_HOLD,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_crisis=vix_crisis,
            vix_calm=vix_calm,
            spike_hold=spike_hold,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_crisis = float(vix_crisis)
        self.vix_calm = float(vix_calm)
        self.spike_hold = int(spike_hold)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self._recovery_start: int | None = None

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Get current VIX
        vix_val = float("nan")
        try:
            vix_hist = ctx.history(_VIX)
            if len(vix_hist) >= 1:
                vix_val = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        # SPY trend for overall bear avoidance
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            spy_hist = None

        spy_bull = True
        if spy_hist is not None and len(spy_hist) >= self.trend_window:
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_now = float(spy_close.iloc[-1])
                spy_bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market - hold TLT defensively
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not np.isfinite(vix_val):
            # No VIX data - default to SPY
            if _SPY in live:
                target[_SPY] = self.exposure
        elif vix_val > self.vix_crisis:
            # Fear peak: start/extend recovery hold
            self._recovery_start = ctx.idx
            # Hold TLT during the actual spike (wait for recovery)
            if _TLT in live:
                target[_TLT] = self.exposure
        elif (self._recovery_start is not None and
              ctx.idx - self._recovery_start <= self.spike_hold):
            # Recovery window: hold QQQ aggressively (post-spike rebound)
            if _QQQ in live:
                target[_QQQ] = self.exposure
        elif vix_val < self.vix_calm:
            # Calm/complacency: QQQ (high-growth environment)
            if _QQQ in live:
                target[_QQQ] = self.exposure
        else:
            # Normal regime: SPY
            if _SPY in live:
                target[_SPY] = self.exposure

        orders: list[Order] = []

        # Liquidate positions not in target
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


NAME = "vix_spike_qqq_recovery"
HYPOTHESIS = (
    "VIX 3-tier regime: VIX>30 hold TLT (spike), then QQQ for 15 bars post-spike "
    "(recovery alpha); VIX<18 hold QQQ (complacency/growth); VIX 18-30 hold SPY; "
    "SPY 200d SMA bear gate to TLT; weekly rebalance; exploits post-crisis equity rebound"
)

STRATEGY = VixSpikeQQQRecovery()
