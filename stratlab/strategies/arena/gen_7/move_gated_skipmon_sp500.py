"""SP500 skip-month momentum gated by ^MOVE (bond volatility) + SPY trend.

Hypothesis: hold top-20 SP500 stocks by 126d-skip-21d (6-1 month skip-month)
momentum when SPY above 100d SMA AND ^MOVE below its 60d MA (bond-vol calm).
Inverse-vol weighted. IEF+GLD 60/37 defensive when either gate fails.
Rebalance every 21 bars (monthly).

Rationale:
  - Skip-month momentum (126d return skipping most recent 21d): the academic
    Jegadeesh-Titman 12-1 momentum factor that avoids short-term reversal
  - ^MOVE gate: the ICE BofA MOVE index measures bond market volatility.
    When MOVE > its 60d MA, bond markets are volatile → usually precedes
    equity stress (2008, 2011, 2020 correlate with MOVE spikes). This gate
    is entirely different from VIX (equity vol) and JNK (credit spreads).
  - SPY 100d SMA (shorter than 200d): captures trend more responsively
  - IEF+GLD defensive: IEF (intermediate bond) + GLD (gold) provides both
    duration exposure and inflation protection in stressed/uncertain regimes

Distinction from existing strategies:
  - ^MOVE as primary volatility gate (no other leaderboard strategy uses MOVE)
  - Skip-month momentum (126d-skip-21d): similar to gen6_credit_gated_skipmon
    but with MOVE gate instead of JNK/LQD credit gate + IEF+GLD defensive
  - SPY 100d SMA (not 200d): faster trend filter
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21    # monthly
MOM_LOOKBACK = 126      # ~6 months
MOM_SKIP = 21           # skip most recent month
VOL_WINDOW = 20         # inverse-vol weights
SPY_TREND_WINDOW = 100  # SPY 100d SMA
MOVE_MA_WINDOW = 60     # MOVE 60d MA
TOP_K = 20
EXPOSURE = 0.97
_MOVE = "^MOVE"


class MoveGatedSkipMonSP500(Strategy):
    """SP500 skip-month momentum gated by SPY 100d SMA + ^MOVE 60d MA.
    IEF+GLD defensive. Inverse-vol weighted. Monthly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        move_ma_window: int = MOVE_MA_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            move_ma_window=move_ma_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.move_ma_window = int(move_ma_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.move_ma_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Gate 1: SPY 100d SMA trend
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.spy_trend_window:
                spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # Gate 2: ^MOVE below 60d MA (bond-vol calm)
        move_calm = False
        try:
            move_hist = ctx.history(_MOVE)
            move_close = move_hist["close"].dropna()
            if len(move_close) >= self.move_ma_window + 1:
                move_current = float(move_close.iloc[-1])
                move_ma = float(move_close.iloc[-self.move_ma_window:].mean())
                move_calm = move_current < move_ma  # MOVE below 60d MA = calm
        except Exception:
            # If MOVE data unavailable, default to calm (permissive)
            move_calm = True

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        risk_on = spy_bull and move_calm

        if not risk_on:
            # Defensive: IEF 60% + GLD 37%
            for sym, w in [("IEF", 0.60), ("GLD", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure / 0.97  # will re-normalize
            total = sum(target.values())
            if total > 0:
                for sym in target:
                    target[sym] = target[sym] / total * self.exposure
            elif "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K SP500 by 126d-skip-21d momentum, inverse-vol weighted
            need = self.mom_lookback + self.vol_window + 10
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_lookback + 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.mom_lookback + 5:
                        continue

                    # Skip-month momentum: 126d return skipping most recent 21d
                    if len(col) < self.mom_lookback + self.mom_skip:
                        continue
                    p_end = float(col.iloc[-(self.mom_skip + 1)])  # 21 bars ago
                    p_start = float(col.iloc[-(self.mom_lookback)])  # 126 bars ago
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    # Inverse-vol weighting (using recent vol)
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    if "IEF" in closes_now.index:
                        target["IEF"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        return []
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["IEF", "GLD", "TLT", "SPY", _MOVE]


NAME = "move_gated_skipmon_sp500"
HYPOTHESIS = (
    "SP500 top-20 by 126d momentum (skip 21d) when SPY above 100d SMA "
    "AND ^MOVE below its 60d MA (bond-vol calm = equity supportive); "
    "inverse-vol weighted; IEF+GLD 60/37 defensive when either gate fails; "
    "rebalance every 21 bars; MOVE gate is distinct from VIX/JNK/credit gates"
)

UNIVERSE = _universe

STRATEGY = MoveGatedSkipMonSP500()
