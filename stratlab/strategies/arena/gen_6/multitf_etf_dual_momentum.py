"""Multi-timeframe dual-momentum ETF rotation strategy.

Hypothesis: Each week rank SPY/QQQ/IWM/EEM/TLT/GLD/IAU by composite
1m+3m+6m return score. Hold top-2 that have ALL timeframes positive
(absolute momentum filter). Hold TLT if only long-term momentum positive.
Hold IEF if TLT also negative. Universe spans equity/bond/EM/gold.

Rationale:
  - Multi-timeframe composite momentum reduces noise from single-window ranking.
  - The "all timeframes positive" absolute filter avoids holding reversing assets
    that show strong 6m momentum but recent weakness.
  - TLT/IEF tiered fallback provides a graceful exit for risk-off regimes.
  - The ETF universe (vs SP500 stocks) produces low correlation with the
    existing stock-level momentum strategies on the leaderboard.

Distinctions from existing leaderboard strategies:
  - gen5_halloween_sell_in_may: seasonal only, not momentum-based.
  - gen5_bond_equity_regime: uses TLT/SPY ratio for regime, not multi-tf momentum.
  - Multi-asset (EEM, GLD/IAU) provides cross-asset diversification signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Lookback windows (trading days)
W1 = 21    # ~1 month
W3 = 63    # ~3 months
W6 = 126   # ~6 months

REBALANCE_EVERY = 5   # weekly

TOP_K = 2
EXPOSURE = 0.97

TRADEABLE = ["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD", "IAU", "IEF"]
SIGNALS: list[str] = []   # no signal-only symbols needed


class MultiTFEtfDualMomentum(Strategy):
    """Multi-timeframe ETF dual momentum: top-2 with all timeframes positive.
    Fallback to TLT → IEF when momentum is missing.
    """

    def __init__(
        self,
        w1: int = W1,
        w3: int = W3,
        w6: int = W6,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            w1=w1,
            w3=w3,
            w6=w6,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.w1 = int(w1)
        self.w3 = int(w3)
        self.w6 = int(w6)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.w6 + 10
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

        # Fetch price histories and compute multi-timeframe momentum scores
        need = self.w6 + 2
        prices_win = ctx.closes_window(need)

        scores: dict[str, float] = {}
        eligible: list[str] = []

        for sym in TRADEABLE:
            if sym not in prices_win.columns:
                continue
            col = prices_win[sym].dropna()
            if len(col) < self.w6:
                continue
            p_now = float(col.iloc[-1])
            if p_now <= 0 or not np.isfinite(p_now):
                continue

            # Compute returns at each timeframe
            if len(col) < self.w1:
                continue
            p_w1 = float(col.iloc[-self.w1])
            if len(col) < self.w3:
                continue
            p_w3 = float(col.iloc[-self.w3])
            p_w6 = float(col.iloc[-self.w6])

            r1 = p_now / p_w1 - 1.0
            r3 = p_now / p_w3 - 1.0
            r6 = p_now / p_w6 - 1.0

            if not (np.isfinite(r1) and np.isfinite(r3) and np.isfinite(r6)):
                continue

            # Composite score (equal-weighted average of the three)
            composite = (r1 + r3 + r6) / 3.0
            scores[sym] = composite

            # Track if all timeframes are positive (absolute momentum filter)
            if r1 > 0 and r3 > 0 and r6 > 0:
                eligible.append(sym)

        target: dict[str, float] = {}

        if eligible:
            # Pick top-K by composite score among those with all-positive momentum
            eligible_scored = {s: scores[s] for s in eligible if s in scores}
            k = min(self.top_k, len(eligible_scored))
            if k > 0:
                ranked = sorted(eligible_scored, key=eligible_scored.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight
        else:
            # No ETF has all-positive momentum: check TLT fallback
            tlt_score = scores.get("TLT", float("-inf"))
            ief_score = scores.get("IEF", float("-inf"))

            if tlt_score > 0 and "TLT" in scores:
                target["TLT"] = self.exposure
            elif ief_score > 0 and "IEF" in scores:
                target["IEF"] = self.exposure
            else:
                # Full cash — hold nothing (stay in cash / existing positions wind down)
                pass

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


UNIVERSE = TRADEABLE  # small explicit list

NAME = "multitf_etf_dual_momentum"
HYPOTHESIS = (
    "Multi-timeframe dual-momentum ETF rotation: each week rank SPY/QQQ/IWM/EEM/"
    "TLT/GLD/IAU by composite 1m+3m+6m return score; hold top-2 with all timeframes "
    "positive; hold TLT if none qualify; IEF if TLT also negative; universe captures "
    "equity/bond/EM/gold diversification"
)

STRATEGY = MultiTFEtfDualMomentum()
