"""Credit-conditioned SP500 momentum strategy.

Hypothesis: Use JNK/LQD 20-day return as a credit spread regime signal.
When JNK outperforms LQD (tightening spreads, risk-on), hold top-20 SP500
stocks by 42-day momentum. When LQD outperforms JNK (widening spreads,
risk-off), rotate to TLT 50% + GLD 50%. Rebalance every 10 bars.

Rationale:
  - Credit spread tightening (HY outperforming IG) signals economic health
    and risk appetite. Using individual stock momentum in risk-on phase
    captures the equity premium more efficiently than a single ETF.
  - 42d momentum window is short enough to respond to regime changes but
    long enough to avoid noise.
  - GLD as 50% of defensive allocation provides inflation protection;
    TLT provides duration/flight-to-quality.

Key distinctions from existing leaderboard:
  - gen5_bond_equity_regime: uses TLT/SPY ratio as regime signal (trend-based),
    not credit spreads (fundamental risk appetite).
  - gen5_credit_spread_hyg_lqd: JNK/LQD MA crossover, holds only HYG or LQD
    (bond-only rotation, no SP500 stock selection).
  - gen5_vix_gated_sp500_momentum: VIX as regime gate (volatility signal),
    not credit spread (fundamental signal).
  - 42d vs 63d momentum window distinguishes from vix_gated (63d).
  - TLT+GLD defensive vs SHY+TLT or TLT-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars
MOMENTUM_WINDOW = 42      # ~2 months for stock selection
CREDIT_WINDOW = 20        # JNK/LQD 20d return as regime signal
TOP_K = 20
EXPOSURE = 0.97

_JNK = "JNK"
_LQD = "LQD"
_TLT = "TLT"
_GLD = "GLD"


class CreditConditionedSP500Momentum(Strategy):
    """Credit-spread-gated SP500 momentum with GLD+TLT defensive bucket."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        credit_window: int = CREDIT_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            credit_window=credit_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.credit_window = int(credit_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.credit_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Compute credit regime: JNK vs LQD 20d return
        risk_on = True  # default to risk-on
        try:
            jnk_hist = ctx.history(_JNK)
            lqd_hist = ctx.history(_LQD)
            if (
                jnk_hist is not None
                and lqd_hist is not None
                and len(jnk_hist) >= self.credit_window + 1
                and len(lqd_hist) >= self.credit_window + 1
            ):
                jnk_close = jnk_hist["close"].dropna()
                lqd_close = lqd_hist["close"].dropna()
                if len(jnk_close) >= self.credit_window and len(lqd_close) >= self.credit_window:
                    jnk_ret = float(jnk_close.iloc[-1] / jnk_close.iloc[-self.credit_window] - 1.0)
                    lqd_ret = float(lqd_close.iloc[-1] / lqd_close.iloc[-self.credit_window] - 1.0)
                    if np.isfinite(jnk_ret) and np.isfinite(lqd_ret):
                        # Risk-on when JNK outperforms LQD (spreads tightening)
                        risk_on = jnk_ret > lqd_ret
        except (KeyError, IndexError):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not risk_on:
            # Risk-off: TLT 50% + GLD 50%
            for sym, w in [(_TLT, 0.5), (_GLD, 0.5)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: top-K SP500 stocks by 42d momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                # Fall back to SPY if too few SP500 stocks
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_JNK, _LQD, _TLT, _GLD, "SPY"]


NAME = "credit_conditioned_sp500_momentum"
HYPOTHESIS = (
    "Credit-conditioned SP500 momentum: when JNK/LQD 20d return > 0 (tightening "
    "spreads, risk-on), hold top-20 SP500 stocks by 42d momentum; when negative "
    "(risk-off), hold TLT 50% + GLD 50%; rebalance every 10 bars"
)

UNIVERSE = _universe

STRATEGY = CreditConditionedSP500Momentum()
