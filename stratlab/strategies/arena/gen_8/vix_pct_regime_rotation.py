"""VIX Rolling Percentile Regime Rotation — gen_8 sonnet-6

Hypothesis: Use the VIX's position in its own 252d rolling distribution (percentile rank)
as an adaptive threshold for regime-switching across QQQ/SPY/TLT+GLD.
- VIX below 25th percentile (persistent calm): QQQ 97% (growth-oriented)
- VIX between 25th and 60th percentile (neutral): SPY 97% (broad market)
- VIX above 60th percentile (elevated stress): TLT 60% + GLD 37% (defensive)
Weekly rebalance.

Rationale:
- Adaptive threshold: unlike fixed VIX level (e.g., "VIX>25"), the percentile
  adapts to the prevailing volatility regime. In 2013-2014 when VIX averaged 14,
  a fixed threshold of 20 means "always calm". The rolling percentile captures
  "calm RELATIVE to the current environment".
- 3-tier allocation: three regimes allow differentiated exposure rather than
  binary on/off — QQQ during structural low-vol, SPY during normal vol, bonds+gold
  during stress.
- GLD + TLT blend in stress (not just TLT): gold provides tail hedge distinct
  from pure duration.

Distinction from existing VIX strategies:
- gen5/gen6 VIX strategies use FIXED levels (VIX<20, VIX>25, VIX<22 etc.)
- This uses ROLLING 252d PERCENTILE — adaptive threshold
- Three tiers: QQQ (low-vol) / SPY (neutral) / TLT+GLD (stress) — not binary
- QQQ in low-vol regime (2013-2015 structural low-vol period would concentrate here)
- Different from opus1_qqq_bollinger_vvix which uses Bollinger %b + VVIX
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
VIX_PERCENTILE_WINDOW = 252  # rolling 252d VIX distribution
VIX_LOW_PCT = 0.25        # below 25th pct = calm regime
VIX_HIGH_PCT = 0.60       # above 60th pct = stressed regime
EXPOSURE = 0.97

_VIX = "^VIX"
_QQQ = "QQQ"   # growth — calm regime
_SPY = "SPY"   # broad — neutral regime
_TLT = "TLT"   # duration — stressed regime
_GLD = "GLD"   # gold — stressed regime hedge
_TLT_WEIGHT = 0.60
_GLD_WEIGHT = 0.37


class VixPctRegimeRotation(Strategy):
    """VIX rolling percentile-based 3-tier regime rotation: QQQ/SPY/TLT+GLD."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_percentile_window: int = VIX_PERCENTILE_WINDOW,
        vix_low_pct: float = VIX_LOW_PCT,
        vix_high_pct: float = VIX_HIGH_PCT,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_percentile_window=vix_percentile_window,
            vix_low_pct=vix_low_pct,
            vix_high_pct=vix_high_pct,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_percentile_window = int(vix_percentile_window)
        self.vix_low_pct = float(vix_low_pct)
        self.vix_high_pct = float(vix_high_pct)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.vix_percentile_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get VIX history for percentile computation
        try:
            vix_hist = ctx.history(_VIX)
        except KeyError:
            return []
        if vix_hist is None or len(vix_hist) < self.vix_percentile_window + 2:
            return []

        vix_close = vix_hist["close"].dropna()
        if len(vix_close) < self.vix_percentile_window + 1:
            return []

        # Current VIX value (yesterday's close, since ctx.history is through yesterday)
        vix_now = float(vix_close.iloc[-1])

        # Rolling 252d distribution: take last 252 values including current
        vix_window = vix_close.iloc[-self.vix_percentile_window:].values
        # Compute percentile rank of current VIX within the window
        vix_pct_rank = float(np.mean(vix_window <= vix_now))

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target allocation based on VIX percentile regime
        target: dict[str, float] = {}

        if vix_pct_rank <= self.vix_low_pct:
            # Persistent calm: QQQ (growth/tech leadership)
            if _QQQ in live:
                target[_QQQ] = self.exposure
            elif _SPY in live:
                target[_SPY] = self.exposure
        elif vix_pct_rank >= self.vix_high_pct:
            # Elevated stress: TLT + GLD defensive blend
            if _TLT in live:
                target[_TLT] = _TLT_WEIGHT
            if _GLD in live:
                target[_GLD] = _GLD_WEIGHT
            # If neither available, hold SPY
            if not target and _SPY in live:
                target[_SPY] = self.exposure
        else:
            # Neutral: broad market SPY
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


UNIVERSE = [_QQQ, _SPY, _TLT, _GLD, _VIX]

NAME = "vix_pct_regime_rotation"
HYPOTHESIS = (
    "VIX 252d rolling percentile regime rotation: hold QQQ 97% when VIX below 25th pct "
    "(low-vol expansion); SPY 97% when VIX between 25th-60th pct (neutral); "
    "TLT 60%+GLD 37% when VIX above 60th pct (elevated stress); "
    "weekly rebalance; adaptive VIX threshold adjusts to current regime "
    "unlike fixed VIX level triggers"
)

STRATEGY = VixPctRegimeRotation()
