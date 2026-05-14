"""SP500 Return-Consistency Momentum — gen_8 sonnet-7

Hypothesis: Rank SP500 stocks by a consistency-adjusted momentum score:
  score = 63d_return / std(daily_returns_over_63d)
This is analogous to a rolling Sharpe/Calmar: it rewards stocks with
steady upward drift, not just high-but-volatile price action.

Rationale:
  - Pure price momentum (used by most leaderboard strategies) selects
    high-beta stocks in bull markets, which can be volatile and drawdown-prone.
  - Consistency-adjusted score filters for stocks with lower return
    dispersion — they have genuine trend character rather than just one
    large upside spike.
  - This produces a portfolio distinct from raw-momentum strategies even
    in the same market regime.

Regime gate: SPY 200d SMA (bear -> TLT defensive)
Sizing: inverse-vol weighted (21d realized vol)
Rebalance: every 10 bars (biweekly)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
SCORE_WINDOW = 63          # ~3 months for both return and std calculation
VOL_WINDOW = 21            # for inverse-vol position sizing
TREND_WINDOW = 200         # 200d SMA gate
TOP_K = 15
EXPOSURE = 0.97
MIN_STD = 1e-6             # avoid division by near-zero std
_SPY = "SPY"
_TLT = "TLT"


class SP500ConsistencyMomentum(Strategy):
    """Consistency-adjusted momentum: return / return-std selection on SP500."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        score_window: int = SCORE_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            score_window=score_window,
            vol_window=vol_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.score_window = int(score_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute consistency-adjusted scores
            need = self.score_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.score_window + 2:
                return []

            scores: dict[str, float] = {}
            vols: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _TLT):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.score_window + 1:
                    continue

                daily_rets = col.pct_change().dropna()
                if len(daily_rets) < self.score_window:
                    continue

                # Use last score_window daily returns
                window_rets = daily_rets.iloc[-self.score_window:]
                total_ret = float(col.iloc[-1] / col.iloc[-self.score_window] - 1.0)
                ret_std = float(window_rets.std())
                if ret_std < MIN_STD or not np.isfinite(total_ret):
                    continue

                score = total_ret / ret_std
                if np.isfinite(score):
                    scores[sym] = score

                # Inverse-vol sizing: use vol_window
                if len(daily_rets) >= self.vol_window:
                    rv = float(daily_rets.iloc[-self.vol_window:].std())
                    vols[sym] = max(rv, 1e-6)

            if len(scores) < 5:
                # Not enough candidates
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Inverse-vol weighting
                inv_vols = {}
                for sym in ranked:
                    vol = vols.get(sym, 0.02)
                    inv_vols[sym] = 1.0 / max(vol, 1e-6)
                total_inv = sum(inv_vols.values())
                if total_inv <= 0:
                    # Fall back to equal weight
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight
                else:
                    for sym in ranked:
                        if sym in live:
                            target[sym] = self.exposure * (inv_vols[sym] / total_inv)

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
    return sp500_tickers() + [_TLT, _SPY]


NAME = "sp500_consistency_momentum"
HYPOTHESIS = (
    "SP500 return-consistency momentum: rank SP500 stocks by 63d return divided by "
    "63d return standard deviation (Calmar-like score), hold top-15 above 200d SMA; "
    "inverse-vol weighted; SPY 200d SMA gate; TLT defensive; biweekly rebalance — "
    "selects stocks with steady positive drift not just high-but-volatile momentum"
)

UNIVERSE = _universe

STRATEGY = SP500ConsistencyMomentum()
