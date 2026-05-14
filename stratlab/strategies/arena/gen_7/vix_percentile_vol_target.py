"""VIX percentile-based SPY vol-targeting strategy.

Hypothesis: Instead of using raw VIX thresholds (which vary by regime), compute
the VIX 252d percentile rank. This normalizes the signal: bottom 25th percentile
= calm (hold SPY 95%), middle = moderate (hold SPY 80%), top quartile = stressed
(hold SPY 50% + TLT 47%). Weekly rebalance.

Rationale: VIX levels change over time — what was "elevated" in 2010 differs
from 2018. Using the 252d percentile rank creates a regime-relative signal that
adapts to the current vol environment. This is distinct from raw-VIX-level gates
already on the leaderboard (vix_gated_sp500_momentum uses raw VIX<25 threshold).

Distinction from existing strategies:
  - VIX percentile rank (not raw level) as regime signal
  - Smooth 3-tier allocation (95/80/50+TLT) vs binary on/off
  - Pure SPY holding (no stock picking) — different trade count pattern
  - Weekly rebalance generates sufficient n_trades
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
VIX_LOOKBACK = 252      # 1-year percentile window
P_LOW = 25              # bottom quartile -> high equity
P_HIGH = 75             # top quartile -> low equity + bonds
SPY_HIGH = 0.95
SPY_MID = 0.80
SPY_LOW = 0.50
TLT_STRESSED = 0.47

_VIX = "^VIX"


class VixPercentileVolTarget(Strategy):
    """VIX 252d percentile rank drives SPY allocation: bottom 25% -> 95%, top 25% -> 50%+TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vix_lookback: int = VIX_LOOKBACK,
        p_low: float = P_LOW,
        p_high: float = P_HIGH,
        spy_high: float = SPY_HIGH,
        spy_mid: float = SPY_MID,
        spy_low: float = SPY_LOW,
        tlt_stressed: float = TLT_STRESSED,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vix_lookback=vix_lookback,
            p_low=p_low,
            p_high=p_high,
            spy_high=spy_high,
            spy_mid=spy_mid,
            spy_low=spy_low,
            tlt_stressed=tlt_stressed,
        )
        self.rebalance_every = int(rebalance_every)
        self.vix_lookback = int(vix_lookback)
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.spy_high = float(spy_high)
        self.spy_mid = float(spy_mid)
        self.spy_low = float(spy_low)
        self.tlt_stressed = float(tlt_stressed)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.vix_lookback + 10
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

        # Compute VIX percentile rank over 252d window
        spy_weight = self.spy_mid  # default
        tlt_weight = 0.0

        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= self.vix_lookback:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= self.vix_lookback:
                    # Current VIX
                    current_vix = float(vix_close.iloc[-1])
                    # 252d window of VIX
                    window = vix_close.iloc[-self.vix_lookback:].values
                    # Percentile rank: what fraction of historical VIX are below current
                    pct_rank = float(np.mean(window <= current_vix) * 100)

                    if pct_rank <= self.p_low:
                        # Very calm: max equity
                        spy_weight = self.spy_high
                        tlt_weight = 0.0
                    elif pct_rank >= self.p_high:
                        # Stressed: reduce equity, add bonds
                        spy_weight = self.spy_low
                        tlt_weight = self.tlt_stressed
                    else:
                        # Middle regime
                        spy_weight = self.spy_mid
                        tlt_weight = 0.0
        except Exception:
            pass

        target: dict[str, float] = {}
        if "SPY" in closes_now.index:
            target["SPY"] = spy_weight
        if tlt_weight > 0 and "TLT" in closes_now.index:
            target["TLT"] = tlt_weight

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


NAME = "vix_percentile_vol_target"
HYPOTHESIS = (
    "VIX percentile-based vol-targeting on SPY: compute VIX 252d percentile; "
    "when VIX in bottom 25pct hold SPY at 95%; middle 25-75pct hold 80%; "
    "top quartile hold SPY 50%+TLT 47%; weekly rebalance; "
    "VIX percentile rank more stable than raw level thresholds"
)

UNIVERSE = ["SPY", "TLT", _VIX]

STRATEGY = VixPercentileVolTarget()
