"""SP500 momentum with triple-gate confirmation.

Hypothesis: use 3 independent risk-on signals (SPY trend, credit, breadth) to
create a graduated equity allocation. All three on -> top-20 SP500 by 42d return
equal-weight at 97%. Two on -> SPY 60%+TLT 37%. One or none on -> TLT 97%.
Rebalance every 10 bars.

Gate logic:
  1. SPY trend: SPY above 200d SMA
  2. Credit: JNK above 20d SMA (risk-on credit conditions)
  3. Breadth: RSP 20d return > SPY 20d return (equal-weight outperforms cap-weight;
     signals broad market participation)

Rationale: Each gate captures a different dimension:
  - SPY trend: are prices in an uptrend?
  - JNK credit: are credit spreads supportive?
  - RSP breadth: is the rally broad or narrow?
  All three "green" = high confidence bull regime; graduated exit as gates flip.

This is distinct from all existing leaderboard strategies because:
  - No other strategy uses RSP/SPY breadth as a gate
  - No other strategy uses a 3-gate graduated allocation
  - Avoids the corr filter issue by using very different allocation logic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # bars
MOMENTUM_WINDOW = 42    # ~2 months
TREND_WINDOW = 200      # SPY 200d SMA
JNK_MA_WINDOW = 20      # JNK 20d SMA credit gate
BREADTH_WINDOW = 20     # RSP vs SPY 20d return
TOP_K = 20
EXPOSURE = 0.97


class TripleGateSP500Momentum(Strategy):
    """SP500 momentum with 3-gate graduated allocation.
    All 3 green -> top-20 stocks. 2 green -> SPY+TLT. 1 or 0 -> TLT.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        jnk_ma_window: int = JNK_MA_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            jnk_ma_window=jnk_ma_window,
            breadth_window=breadth_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.jnk_ma_window = int(jnk_ma_window)
        self.breadth_window = int(breadth_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Gate 1: SPY 200d SMA trend
        spy_gate = False
        try:
            spy_hist = ctx.history("SPY")
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_gate = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # Gate 2: JNK above 20d SMA (credit gate)
        jnk_gate = False
        try:
            jnk_hist = ctx.history("JNK")
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma_window + 1:
                jnk_ma = float(jnk_close.iloc[-self.jnk_ma_window:].mean())
                jnk_gate = float(jnk_close.iloc[-1]) > jnk_ma
        except Exception:
            pass

        # Gate 3: RSP 20d return > SPY 20d return (breadth gate)
        breadth_gate = False
        try:
            rsp_hist = ctx.history("RSP")
            rsp_close = rsp_hist["close"].dropna()
            spy_hist2 = ctx.history("SPY")
            spy_close2 = spy_hist2["close"].dropna()
            if len(rsp_close) >= self.breadth_window + 1 and len(spy_close2) >= self.breadth_window + 1:
                rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                spy_ret = float(spy_close2.iloc[-1] / spy_close2.iloc[-self.breadth_window] - 1.0)
                breadth_gate = rsp_ret > spy_ret
        except Exception:
            pass

        gates_on = sum([spy_gate, jnk_gate, breadth_gate])

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if gates_on == 0 or gates_on == 1:
            # Risk-off: TLT only
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif gates_on == 2:
            # Mixed: SPY 60% + TLT 37%
            for sym, w in [("SPY", 0.60), ("TLT", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure / 0.97  # normalize to exposure
            # Re-normalize
            total = sum(target.values())
            if total > 0:
                for sym in target:
                    target[sym] = target[sym] / total * self.exposure
        else:
            # All 3 gates on: top-K SP500 by 42d momentum, equal-weight
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
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

                if len(scores) < 5:
                    if "TLT" in closes_now.index:
                        target["TLT"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", "JNK", "RSP"]


NAME = "triple_gate_sp500_momentum"
HYPOTHESIS = (
    "SP500 top-20 momentum with 3-gate confirmation: SPY above 200d SMA (trend), "
    "JNK above 20d SMA (credit), AND RSP above SPY 20d return (breadth); "
    "all three risk-on -> top-20 SP500 by 42d return equal-weight; "
    "two risk-on -> SPY 60%+TLT 37%; one or none -> TLT 97%; "
    "rebalance every 10 bars; triple confirmation avoids false-positive entries"
)

UNIVERSE = _universe

STRATEGY = TripleGateSP500Momentum()
