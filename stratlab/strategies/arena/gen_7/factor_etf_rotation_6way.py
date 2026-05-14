"""6-way factor ETF rotation strategy.

Hypothesis: rank MTUM/QUAL/SPLV/USMV/IVE/IWB by 63d return; hold top-2
equal-weight; if all have negative 63d return (absolute momentum test) hold TLT;
rebalance every 10 bars.

Rationale: Factor ETFs (momentum, quality, low-vol, value, large-blend) rotate
leadership depending on the macro environment. Unlike the gen6_factor_etf_rotation
(MTUM/QUAL/IVE/USMV), this expands to 6 factors including SPLV (low-vol) and
IWB (Russell 1000 broad market), uses a 63d window vs 3-month composite, adds
an absolute momentum filter (TLT if all negative), and rebalances every 10 bars
vs monthly. The pure factor ETF universe means no individual stock risk and
naturally low correlation to SP500 cross-sectional strategies.

Key distinctions from gen6_factor_etf_rotation:
  - 6 factor ETFs vs 4 (adds SPLV low-vol and IWB broad-market)
  - 63d momentum window (same as many stock strategies but on factor ETFs)
  - Absolute momentum filter: hold TLT if all factor ETFs negative
  - 10-bar rebalance (biweekly) vs monthly — more trades
  - No VIX or credit gate — pure factor rotation signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars
MOMENTUM_WINDOW = 63      # ~3 months
TOP_K = 2
EXPOSURE = 0.97

FACTOR_ETFS = ["MTUM", "QUAL", "SPLV", "USMV", "IVE", "IWB"]
DEFENSIVE = "TLT"


class FactorEtfRotation6Way(Strategy):
    """Top-2 of MTUM/QUAL/SPLV/USMV/IVE/IWB by 63d return; TLT if all negative."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 10
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

        need = self.momentum_window + 5
        prices = ctx.closes_window(need)

        scores: dict[str, float] = {}
        for sym in FACTOR_ETFS:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-self.momentum_window])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            if np.isfinite(ret):
                scores[sym] = ret

        target: dict[str, float] = {}

        if not scores:
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        else:
            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

            # Absolute momentum: if all factor ETFs have negative 63d return, go defensive
            if scores[ranked[0]] <= 0:
                if DEFENSIVE in closes_now.index:
                    target[DEFENSIVE] = self.exposure
            else:
                # Hold top-k with positive or mixed momentum (at least top has positive)
                positive_ranked = [s for s in ranked if scores[s] > 0]
                if not positive_ranked:
                    if DEFENSIVE in closes_now.index:
                        target[DEFENSIVE] = self.exposure
                else:
                    per_wt = self.exposure / len(positive_ranked)
                    for sym in positive_ranked:
                        target[sym] = per_wt

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


UNIVERSE = FACTOR_ETFS + [DEFENSIVE]

NAME = "factor_etf_rotation_6way"
HYPOTHESIS = (
    "Factor ETF rotation MTUM/QUAL/SPLV/USMV/IVE/IWB by 63d return: hold top-2 equal-weight; "
    "when all have negative 63d return hold TLT; rebalance every 10 bars; "
    "6-factor ETF rotation captures style rotation without individual stock risk"
)

STRATEGY = FactorEtfRotation6Way()
