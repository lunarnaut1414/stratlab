"""GLD/SPY/TLT tri-state regime via VIX rate-of-change.

Hypothesis: hold GLD 97% when VIX 5d change > +5 (panic spike, gold as safe haven);
TLT 97% when VIX level > 25 (sustained stress); SPY 97% otherwise; weekly rebalance.
VIX-acceleration as primary signal different from VIX-level gating.

Rationale: VIX-level-based strategies (VIX > 25) are well-represented on the leaderboard.
This strategy uses a different signal: the *acceleration* of VIX — when VIX spikes
suddenly by +5 points in 5 days, gold historically outperforms both equities and bonds
(short-term panic hedge). Sustained high VIX (> 25) keeps TLT as the defensive.
Normal VIX regimes hold SPY for market beta.

Distinction from existing strategies:
  - Uses VIX 5-day rate-of-change, not VIX level, as primary signal
  - GLD as panic-spike hedge (not just TLT or SHY)
  - Pure ETF rotation: SPY/TLT/GLD (no individual stocks)
  - Orthogonal to credit-spread and momentum strategies
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
VIX_SPIKE_LOOKBACK = 5     # days to measure VIX change
VIX_SPIKE_THRESHOLD = 5.0  # VIX points increase triggers gold
VIX_STRESS_LEVEL = 25.0    # sustained VIX above this -> TLT
EXPOSURE = 0.97
_VIX = "^VIX"


class VixRocGoldRegime(Strategy):
    """Tri-state GLD/TLT/SPY regime using VIX rate-of-change and VIX level."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_spike_lookback: int = VIX_SPIKE_LOOKBACK,
        vix_spike_threshold: float = VIX_SPIKE_THRESHOLD,
        vix_stress_level: float = VIX_STRESS_LEVEL,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_spike_lookback=vix_spike_lookback,
            vix_spike_threshold=vix_spike_threshold,
            vix_stress_level=vix_stress_level,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_spike_lookback = int(vix_spike_lookback)
        self.vix_spike_threshold = float(vix_spike_threshold)
        self.vix_stress_level = float(vix_stress_level)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.vix_spike_lookback + 20
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

        # Get VIX history
        vix_current = float("nan")
        vix_prev = float("nan")
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= self.vix_spike_lookback + 1:
                vix_series = vix_hist["close"].dropna()
                if len(vix_series) >= self.vix_spike_lookback + 1:
                    vix_current = float(vix_series.iloc[-1])
                    vix_prev = float(vix_series.iloc[-self.vix_spike_lookback - 1])
        except Exception:
            pass

        # Determine regime
        target: dict[str, float] = {}

        if np.isfinite(vix_current) and np.isfinite(vix_prev):
            vix_change = vix_current - vix_prev

            if vix_change >= self.vix_spike_threshold:
                # Panic spike: GLD as safe haven
                if "GLD" in closes_now.index:
                    target["GLD"] = self.exposure
                elif "IAU" in closes_now.index:
                    target["IAU"] = self.exposure
            elif vix_current >= self.vix_stress_level:
                # Sustained stress: TLT defensive
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Normal regime: SPY
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
        else:
            # No VIX data: default to SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

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


NAME = "vix_roc_gold_regime"
HYPOTHESIS = (
    "GLD/SPY/TLT tri-state regime via VIX rate-of-change: hold GLD 97% when VIX 5d change > +5 "
    "(panic spike, gold as safe haven); TLT 97% when VIX level > 25 (sustained stress); "
    "SPY 97% otherwise; weekly rebalance; VIX-acceleration as primary signal different from VIX-level gating"
)

UNIVERSE = ["SPY", "TLT", "GLD", "IAU", _VIX]

STRATEGY = VixRocGoldRegime()
