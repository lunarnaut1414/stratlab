"""opus-1 mutation of bond_equity_regime (parent IS Calmar 0.64, corr 0.00).

Mutates the regime signal from TLT/SPY MA crossover to a JNK/LQD z-score
breakout — credit-spread risk appetite, three-state regime — with a SP500
21d-skip-1 momentum bucket on the risk-on side and a TLT/GLD safe-haven
bucket on the risk-off side.

  - Ratio:        TLT/SPY            ->  JNK/LQD (high-yield / IG corporate;
                                          credit-spread risk appetite, NOT
                                          bond-vs-equity).
  - Statistic:    50d MA crossover   ->  90d z-score, three states:
                                            z >  +0.5  -> risk-on
                                            z < -0.5   -> risk-off
                                            -0.5..+0.5 -> neutral (always
                                                          invested via 60/40)
  - Risk-on:      top-10 SP500 by    ->  top-10 SP500 by 21d momentum
                  63d momentum            (skip last 1 day).
  - Risk-off:     TLT 60% + GLD 40%  ->  TLT 50% + GLD 50%.
  - Neutral:      n/a                ->  SPY 60% + TLT 40% (always invested).
  - Rebalance:    10 bars            ->  10 bars (same).

JNK (cached 2007-12-) and LQD (2002-07-) both pre-date IS_START so no
yfinance refresh-on-load issue. The three-state structure plus the credit
signal source means daily returns should diverge from the parent's
two-state TLT/SPY-driven path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE = 10
Z_WINDOW = 90
Z_HIGH = 0.5
Z_LOW = -0.5
MOM_LOOKBACK = 21
MOM_SKIP = 1
TOP_K = 10
EXPOSURE = 0.97


class CreditZScoreBreakout(Strategy):
    def __init__(
        self,
        rebalance: int = REBALANCE,
        z_window: int = Z_WINDOW,
        z_high: float = Z_HIGH,
        z_low: float = Z_LOW,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance=rebalance,
            z_window=z_window,
            z_high=z_high,
            z_low=z_low,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance = int(rebalance)
        self.z_window = int(z_window)
        self.z_high = float(z_high)
        self.z_low = float(z_low)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.z_window, self.mom_lookback + self.mom_skip) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        try:
            jnk = ctx.history("JNK")
            lqd = ctx.history("LQD")
        except KeyError:
            return []
        if len(jnk) < self.z_window + 5 or len(lqd) < self.z_window + 5:
            return []

        jnk_c = jnk["close"]
        lqd_c = lqd["close"]
        ratio = (jnk_c / lqd_c).dropna()
        if len(ratio) < self.z_window:
            return []
        tail = ratio.iloc[-self.z_window:]
        mu = float(tail.mean())
        sd = float(tail.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-9:
            return []
        z = (float(ratio.iloc[-1]) - mu) / sd

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if z > self.z_high:
            need = self.mom_lookback + self.mom_skip + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []
            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue
                p_end = float(col.iloc[-self.mom_skip - 1])
                p_start = float(col.iloc[-self.mom_skip - self.mom_lookback - 1])
                if p_start <= 0 or not np.isfinite(p_end) or not np.isfinite(p_start):
                    continue
                scores[sym] = p_end / p_start - 1.0
            if len(scores) < self.top_k:
                return []
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_k]
            per = self.exposure / len(ranked)
            for sym in ranked:
                target[sym] = per
        elif z < self.z_low:
            for sym, w in [("TLT", 0.5), ("GLD", 0.5)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            for sym, w in [("SPY", 0.6), ("TLT", 0.4)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure

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
    return sp500_tickers() + ["JNK", "LQD", "TLT", "GLD", "SPY"]


NAME = "opus1_credit_zscore_breakout"
HYPOTHESIS = (
    "Mutate bond_equity_regime: JNK/LQD z-score breakout (90d window, ±0.5 "
    "thresholds) — z>+0.5 hold top-10 SP500 21d-skip-1 momentum, z<-0.5 hold "
    "TLT+GLD, neutral SPY/TLT 60/40. Three-state credit-spread regime."
)

UNIVERSE = _universe

STRATEGY = CreditZScoreBreakout()
