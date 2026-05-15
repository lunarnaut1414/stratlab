"""Multi-asset ETF cross-sectional momentum with volatility carry.

Hypothesis: Rank popular ETFs by momentum but weight them inversely to their
realized vol — this is a "volatility carry" approach that consistently favors
lower-vol ETFs when their momentum signal is equal, without specifically gating
on any macro signal. Unlike the SP500 stock-pickers on the leaderboard, this
strategy rotates across asset classes (equity, bond, commodity, real estate,
international) using the popular ETF universe.

The mechanism is orthogonal to all existing strategies:
  - NOT SP500 individual stock selection (picks ETFs)
  - NOT a macro-signal gating strategy (no VIX, credit, yield curve gate)
  - NOT sector rotation (uses full cross-asset ETF universe)
  - Pure cross-sectional relative momentum with vol-carry sizing

Design:
  - Universe: popular ETFs (broad_market, sector, bonds, commodities, international)
    that cover the IS window.
  - Rank all ETFs by 63d return (cross-sectional momentum).
  - Hold top-8 ETFs with positive absolute momentum (>0% 63d return).
  - If fewer than 3 qualify, hold SHY as defensive.
  - Weight inversely by 21d realized vol (vol-carry).
  - Rebalance every 10 bars (biweekly).
  - No macro gate (no VIX, no credit spread) — pure cross-section.
  - Exclude short/inverse ETFs and leveraged ETFs from tradeable set.

This is the closest equivalent to a "pure factor" strategy in the ETF space.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # 3-month cross-sectional momentum
MIN_ABS_MOM = 0.0         # require positive absolute momentum (>0%)
VOL_WINDOW = 21           # for inverse-vol carry weight
MIN_POSITIONS = 3         # minimum qualifying ETFs before going defensive
TOP_K = 8                 # hold top-K ETFs
EXPOSURE = 0.97

# ETFs known to have IS coverage (inception before 2010, broad asset class coverage)
TRADEABLE_ETFS = [
    # Broad equity
    "SPY", "QQQ", "IWM", "IWF", "IWD", "MDY",
    # Sector ETFs
    "XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLE",
    # International
    "EFA", "EEM",
    # Bonds
    "TLT", "IEF", "SHY", "LQD", "HYG",
    # Commodities / Real assets
    "GLD", "SLV", "USO", "DBC",
    # Real Estate
    "VNQ",
    # Small/value
    "IWN",
]
DEFENSIVE_ETF = "SHY"


class ETFXSectVolCarry(Strategy):
    """Popular ETF cross-sectional 63d momentum with inverse-vol carry weighting;
    top-8 ETFs with positive absolute momentum; SHY defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        min_abs_mom: float = MIN_ABS_MOM,
        vol_window: int = VOL_WINDOW,
        min_positions: int = MIN_POSITIONS,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            min_abs_mom=min_abs_mom,
            vol_window=vol_window,
            min_positions=min_positions,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.min_abs_mom = float(min_abs_mom)
        self.vol_window = int(vol_window)
        self.min_positions = int(min_positions)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.vol_window + 10
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

        need = self.momentum_window + self.vol_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < need - 10:
            return []

        scores: dict[str, float] = {}
        inv_vols: dict[str, float] = {}

        for sym in TRADEABLE_ETFS:
            if sym == DEFENSIVE_ETF:
                continue
            if sym not in prices.columns:
                continue

            col = prices[sym].dropna()
            if len(col) < need - 10:
                continue

            arr = col.values

            # 63d momentum
            if len(arr) < self.momentum_window + 2:
                continue
            p_end = float(arr[-1])
            p_start = float(arr[-self.momentum_window])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            if not np.isfinite(ret) or ret <= self.min_abs_mom:
                continue

            # Inverse-vol weight
            if len(arr) < self.vol_window + 1:
                continue
            tail = arr[-(self.vol_window + 1):]
            logr = np.log(tail[1:] / tail[:-1])
            rv = float(np.std(logr))
            if rv <= 1e-6 or not np.isfinite(rv):
                continue

            scores[sym] = ret
            inv_vols[sym] = 1.0 / rv

        target: dict[str, float] = {}

        if len(scores) < self.min_positions:
            # Defensive: not enough qualifying ETFs
            if DEFENSIVE_ETF in closes_now.index:
                target[DEFENSIVE_ETF] = self.exposure
        else:
            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
            iv_sum = sum(inv_vols[s] for s in ranked)
            if iv_sum <= 0:
                return []
            for sym in ranked:
                target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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


NAME = "etf_xsect_vol_carry"
HYPOTHESIS = (
    "Multi-asset ETF cross-sectional 63d momentum with inverse-vol carry weighting: rank popular "
    "ETFs across equity/bond/commodity/international by 63d return; hold top-8 with positive "
    "absolute momentum; weight inversely by 21d realized vol (vol-carry); SHY defensive when "
    "<3 qualify; biweekly rebalance; no macro gate — pure cross-section on ETF universe; "
    "orthogonal to all SP500 stock-selection strategies on leaderboard"
)

UNIVERSE = list(set(TRADEABLE_ETFS + [DEFENSIVE_ETF]))

STRATEGY = ETFXSectVolCarry()
