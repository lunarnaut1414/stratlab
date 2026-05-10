"""Breadth-gated SP500 momentum strategy.

Hypothesis: Use the fraction of SP500 stocks with positive 10-day returns
as a "breadth" regime signal.
  - Breadth ≥ 60% (broad participation): hold top-20 SP500 stocks by 63d
    skip-5d momentum, equally weighted.
  - Breadth 40-60% (mixed): hold SPY (neutral, broad exposure).
  - Breadth < 40% (narrow/deteriorating): hold TLT (defensive).
  Weekly rebalance (every 5 bars).

Rationale:
  Market breadth — the fraction of stocks rising — measures the quality and
  sustainability of a trend. A rally where 80% of stocks participate is
  more durable than one led by 5 mega-caps. When breadth is high, stock
  momentum has more names to work with and lower crash risk. When breadth
  drops below 40%, it signals distribution and portfolio protection is needed.

  The 63d skip-5d momentum (skipping only 1 week) is a slight variant of
  standard 63d momentum that avoids the shortest-term reversal while
  still being responsive to recent 3-month trends.

Diversification vs leaderboard:
  - No VIX gate, no credit spread, no TLT/SPY ratio — breadth IS the signal.
  - The breadth regime is not used in any existing gen_5 or gen_6 accepted
    strategy. All existing strategies use VIX levels, SMA crossovers, or
    credit spreads.
  - RSP/SPY breadth proxy (gen5_atr_momentum_etf) used RSP/SPY ratio MA;
    this uses actual cross-sectional breadth % — a different calculation.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
MOMENTUM_WINDOW = 63     # ~3 months
MOM_SKIP = 5             # skip last 1 week (short-term reversal avoidance)
BREADTH_WINDOW = 10      # 10d return for breadth calculation
TOP_K = 20
BREADTH_HIGH = 0.60      # above this -> full momentum
BREADTH_LOW = 0.40       # below this -> TLT
EXPOSURE = 0.97


class BreadthGatedSP500Momentum(Strategy):
    """Breadth-gated momentum: top-20 SP500 by 63d skip-5d momentum when breadth >= 60%."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        mom_skip: int = MOM_SKIP,
        breadth_window: int = BREADTH_WINDOW,
        top_k: int = TOP_K,
        breadth_high: float = BREADTH_HIGH,
        breadth_low: float = BREADTH_LOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            mom_skip=mom_skip,
            breadth_window=breadth_window,
            top_k=top_k,
            breadth_high=breadth_high,
            breadth_low=breadth_low,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.mom_skip = int(mom_skip)
        self.breadth_window = int(breadth_window)
        self.top_k = int(top_k)
        self.breadth_high = float(breadth_high)
        self.breadth_low = float(breadth_low)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.mom_skip + 10
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

        # Calculate breadth: fraction of stocks with positive 10d return
        need = self.momentum_window + self.mom_skip + 5
        prices = ctx.closes_window(need)
        if len(prices) < self.breadth_window + 2:
            return []

        n_positive = 0
        n_total = 0
        for sym in prices.columns:
            # Skip ETFs/bonds from breadth calculation — only measure stock breadth
            if sym in {"TLT", "GLD", "SHY", "IEF", "AGG", "JNK", "LQD",
                       "SPY", "QQQ", "IWM", "DBC", "SSO", "TQQQ", "GDX",
                       "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY",
                       "RSP", "HYG", "SMH", "SHV", "BIL"}:
                continue
            col = prices[sym].dropna()
            if len(col) < self.breadth_window + 1:
                continue
            ret_10d = float(col.iloc[-1] / col.iloc[-(self.breadth_window + 1)] - 1.0)
            if np.isfinite(ret_10d):
                n_total += 1
                if ret_10d > 0:
                    n_positive += 1

        if n_total < 50:  # not enough stocks to measure breadth
            breadth = 0.5  # neutral
        else:
            breadth = n_positive / n_total

        target: dict[str, float] = {}

        if breadth < self.breadth_low:
            # Narrow breadth / deteriorating: defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif breadth < self.breadth_high:
            # Mixed breadth: neutral SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            # Broad participation: stock momentum
            if len(prices) < self.momentum_window + self.mom_skip:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in {"TLT", "GLD", "SHY", "IEF", "AGG", "JNK", "LQD",
                               "SPY", "QQQ", "IWM", "DBC", "SSO", "TQQQ", "GDX",
                               "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY",
                               "RSP", "HYG", "SMH", "SHV", "BIL"}:
                        continue
                    col = prices[sym].dropna()
                    total_needed = self.momentum_window + self.mom_skip
                    if len(col) < total_needed + 1:
                        continue
                    # Skip-5d momentum
                    p_end = float(col.iloc[-(self.mom_skip + 1)])
                    p_start = float(col.iloc[-(total_needed + 1)])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:self.top_k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "breadth_gated_sp500_momentum"
HYPOTHESIS = (
    "SP500 breadth-gated momentum: calculate fraction of SP500 stocks with positive 10d "
    "return; when breadth >= 60% hold top-20 by 63d skip-5d momentum; breadth 40-60% hold "
    "SPY; breadth < 40% hold TLT; weekly rebalance; breadth regime signal is orthogonal "
    "to VIX/credit/trend signals already on leaderboard."
)

UNIVERSE = _universe

STRATEGY = BreadthGatedSP500Momentum()
