"""Absolute + Relative Momentum on Multi-Asset ETF Basket (Antonacci-style).

Hypothesis:
  Each month, rank SPY, EEM, TLT, GLD, DBC by 12-month total return.
  For the top-ranked asset, apply absolute momentum check:
    if its own 12m return > 0: hold it at 97% exposure.
    if its own 12m return <= 0 (negative absolute momentum): hold SHY (cash).
  Rebalance monthly (every 21 bars).

Rationale:
  Gary Antonacci's Global Momentum (GEM) methodology: always hold the best
  cross-asset performer, but only if it has positive absolute momentum.
  This is structurally distinct from all existing leaderboard strategies:
  - No VIX gating (uses pure return-based momentum)
  - No SP500 single-stock selection (all ETFs, cross-asset)
  - No sector rotation or mean-reversion
  - Absolute momentum check prevents holding assets in structural downtrends

  Key difference from gen5_multiasset_abs_momentum (which failed):
  - Focus on just 5 best-known assets (less noise)
  - Top-1 winner-takes-all (higher concentration, higher momentum capture)
  - Keep SHY as cash proxy (not TLT which underperformed in rising-rate 2013-2018)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

CANDIDATES = ["SPY", "EEM", "TLT", "GLD", "DBC"]
LOOKBACK = 252          # 12-month momentum window
REBALANCE_EVERY = 21    # monthly
EXPOSURE = 0.97
CASH_PROXY = "SHY"


class AbsoluteMomentumETF(Strategy):
    """Antonacci-style absolute + relative momentum on 5 major ETF classes."""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            lookback=lookback,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.lookback = int(lookback)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.lookback + 10
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

        # Compute 12-month returns for candidates
        prices = ctx.closes_window(self.lookback + 5)
        if len(prices) < self.lookback:
            return []

        scores: dict[str, float] = {}
        for sym in CANDIDATES:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.lookback:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-self.lookback])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            scores[sym] = ret

        if not scores:
            return []

        # Select the best-performing asset
        best = max(scores, key=scores.__getitem__)
        best_ret = scores[best]

        # Absolute momentum check: if best is negative, go to cash
        if best_ret <= 0.0:
            chosen = CASH_PROXY
        else:
            chosen = best

        if chosen not in closes_now.index:
            chosen = CASH_PROXY
        if chosen not in closes_now.index:
            return []

        target: dict[str, float] = {chosen: self.exposure}

        # Build orders
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


NAME = "absolute_momentum_etf"
HYPOTHESIS = (
    "Multi-asset absolute+relative momentum (Antonacci GEM style): rank SPY, EEM, TLT, GLD, DBC "
    "by 12-month return; hold the top asset if its 12m return > 0; otherwise hold SHY (cash). "
    "Monthly rebalance. Absolute momentum gate prevents holding assets in structural downtrends."
)

UNIVERSE = CANDIDATES + [CASH_PROXY]

STRATEGY = AbsoluteMomentumETF()
