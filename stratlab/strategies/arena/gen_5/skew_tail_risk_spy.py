"""SKEW Tail-Risk Gated SPY Allocator — gen_5 opus-2 (gap_finder)

Hypothesis: ^SKEW (CBOE Skew Index) measures the perceived tail-risk
priced into S&P 500 OTM options. SKEW>140 historically marks elevated
tail risk priced by options traders. As SKEW rises, equity allocation
should be trimmed in favor of safer assets, even when VIX itself is
calm. SKEW captures *what's missing* from VIX — the asymmetric
left-tail probability density.

Allocation:
  - SKEW 20d MA <= 125: SPY 95% (calm tail)
  - 125 < SKEW 20d MA <= 140: SPY 70% + IEF 30% (mild tail risk)
  - SKEW 20d MA > 140: SPY 30% + IEF 65% (elevated tail risk)

Gap addressed: leaderboard has zero strategies using ^SKEW. ^SKEW is
literally orthogonal to VIX — VIX is the volatility expectation, SKEW
is the tail asymmetry.

^SKEW data: 1990-01-02 onwards — full IS coverage.

IS window: 2010-01-01 to 2018-12-31.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "IEF", "SHY", "^SKEW"]

_SKEW = "^SKEW"
_MA_WINDOW = 20
_LOW_THR = 125.0
_HIGH_THR = 140.0
_REBALANCE = 5


class SkewTailRiskSpy(Strategy):
    """SPY allocator gated by ^SKEW 20d MA tiers."""

    def __init__(
        self,
        ma_window: int = _MA_WINDOW,
        low_thr: float = _LOW_THR,
        high_thr: float = _HIGH_THR,
        rebalance: int = _REBALANCE,
    ) -> None:
        super().__init__(
            ma_window=ma_window,
            low_thr=low_thr,
            high_thr=high_thr,
            rebalance=rebalance,
        )
        self.ma_window = ma_window
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.rebalance = rebalance

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.ma_window + 5:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        try:
            skew_hist = ctx.history(_SKEW)
        except KeyError:
            return []
        if skew_hist is None or len(skew_hist) < self.ma_window:
            return []

        skew_close = skew_hist["close"].astype(float).dropna()
        if len(skew_close) < self.ma_window:
            return []

        skew_ma = float(skew_close.iloc[-self.ma_window:].mean())

        # Tier-based allocation
        if skew_ma <= self.low_thr:
            target_weights = {"SPY": 0.95}
        elif skew_ma <= self.high_thr:
            target_weights = {"SPY": 0.70, "IEF": 0.27}
        else:
            target_weights = {"SPY": 0.30, "IEF": 0.65}

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        # Filter target_weights to those with prices
        target = {s: w for s, w in target_weights.items() if s in closes_now.index}

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "skew_tail_risk_spy"
HYPOTHESIS = (
    "SKEW tail-risk gated SPY allocator: SPY 95% when ^SKEW 20d MA below 125; "
    "SPY 70% + IEF 27% in 125-140; SPY 30% + IEF 65% above 140; pure ^SKEW "
    "tail-risk regime (orthogonal to VIX)."
)

STRATEGY = SkewTailRiskSpy()
