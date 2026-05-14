"""QQQ + SPY Breadth-Credit Composite — gen_7 sonnet-7 (attempt 9)

Hypothesis: Use a composite signal of two orthogonal risk-on indicators:
  1. RSP/SPY 30d return spread (broad market breadth)
  2. JNK 30d return vs 90d return (credit acceleration)

When BOTH signals are positive (bull-on-bull): hold QQQ 97%
When breadth positive but credit decelerating: hold SPY 97%
When credit positive but breadth failing: hold SPY 60% + TLT 37%
When BOTH negative: hold TLT 60% + SHY 37%

Rationale: RSP (equal-weight SP500) outperforming SPY (cap-weight) indicates
broad market participation, not just mega-cap leadership. JNK acceleration
(recent vs slower momentum) confirms credit conditions improving. Both signals
together provide confirmation of risk-on before deploying into QQQ.

This is distinct from jnk_vix_dual_gate_qqq (uses level-based VIX not breadth)
and from smallcap_leadership_rotation (different breadth measure and 4-tier logic).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "SHY", "RSP", "JNK"]

REBALANCE_EVERY = 5        # weekly
BREADTH_WINDOW = 30        # RSP vs SPY 30d return comparison
JNK_FAST = 20              # JNK fast momentum window
JNK_SLOW = 60              # JNK slow momentum window
TREND_WINDOW = 200
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_SHY = "SHY"
_RSP = "RSP"
_JNK = "JNK"


class QQQBreadthCreditRegime(Strategy):
    """Composite breadth+credit regime: QQQ when both bullish; SPY/TLT/SHY otherwise."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_window: int = BREADTH_WINDOW,
        jnk_fast: int = JNK_FAST,
        jnk_slow: int = JNK_SLOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            jnk_fast=jnk_fast,
            jnk_slow=jnk_slow,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.jnk_fast = int(jnk_fast)
        self.jnk_slow = int(jnk_slow)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_slow, self.trend_window) + 10
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

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            # Bear: TLT+SHY defensive
            if _TLT in live:
                target[_TLT] = 0.60 * self.exposure
            if _SHY in live:
                target[_SHY] = 0.37 * self.exposure
        else:
            # Compute signals
            need = self.jnk_slow + 5
            prices = ctx.closes_window(need)

            # Signal 1: RSP/SPY breadth
            breadth_bull = False
            if (_RSP in prices.columns and _SPY in prices.columns and
                    len(prices) >= self.breadth_window):
                rsp_col = prices[_RSP].dropna()
                spy_col = prices[_SPY].dropna()
                min_len = min(len(rsp_col), len(spy_col))
                if min_len >= self.breadth_window:
                    rsp_ret = float(rsp_col.iloc[-1] / rsp_col.iloc[-self.breadth_window] - 1.0)
                    spy_ret = float(spy_col.iloc[-1] / spy_col.iloc[-self.breadth_window] - 1.0)
                    breadth_bull = rsp_ret > spy_ret

            # Signal 2: JNK acceleration (fast > slow momentum)
            credit_bull = False
            if _JNK in prices.columns and len(prices) >= self.jnk_slow:
                jnk_col = prices[_JNK].dropna()
                if len(jnk_col) >= self.jnk_slow:
                    jnk_fast_ret = float(jnk_col.iloc[-1] / jnk_col.iloc[-self.jnk_fast] - 1.0)
                    jnk_slow_ret = float(jnk_col.iloc[-1] / jnk_col.iloc[-self.jnk_slow] - 1.0)
                    # Credit accelerating: recent JNK momentum stronger than longer-term
                    credit_bull = jnk_fast_ret > 0 and jnk_fast_ret > jnk_slow_ret / 2

            if breadth_bull and credit_bull:
                # Both bullish: QQQ
                if _QQQ in live:
                    target[_QQQ] = self.exposure
            elif breadth_bull:
                # Breadth OK, credit decelerating: SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            elif credit_bull:
                # Credit OK but breadth failing: SPY/TLT split
                if _SPY in live:
                    target[_SPY] = 0.60 * self.exposure
                if _TLT in live:
                    target[_TLT] = 0.37 * self.exposure
            else:
                # Both bearish: TLT+SHY
                if _TLT in live:
                    target[_TLT] = 0.60 * self.exposure
                if _SHY in live:
                    target[_SHY] = 0.37 * self.exposure

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


NAME = "qqq_breadth_credit_regime"
HYPOTHESIS = (
    "QQQ breadth+credit composite regime: RSP>SPY 30d (broad breadth) AND JNK 20d>60d "
    "acceleration (credit momentum) -> QQQ 97%; breadth only -> SPY; credit only -> "
    "SPY+TLT; neither -> TLT+SHY; SPY 200d SMA bear gate; weekly rebalance; orthogonal "
    "composite signal vs single-signal JNK or VIX gating strategies"
)

STRATEGY = QQQBreadthCreditRegime()
