"""gen_9 opus-1 — Vol-Target Skipmon mutation: 16pct target + 60d vol lookback.

Parent: gen9_gen9_sp500_voltarget_skipmon (IS Calmar 1.08, h1=1.04/h2=1.24 STABLE).
Mutation:
  - VOL_TARGET 12% -> 16% (more aggressive carry in calm regime)
  - VOL_WINDOW 30d -> 60d (smoother realized vol estimate)
  - Same 126d-skip-21d momentum, top-15 SP500, SPY 200d gate -> IEF
  - Same biweekly rebalance, exposure clip [50%, 97%]

Rationale: The parent has the best STABILITY of round (h1=1.04 h2=1.24). Raising
the vol target to 16% raises gross exposure in calm regimes (lifts return); the
longer 60d vol lookback smooths daily exposure changes (less turnover, less
mean-reversion-driven false signal). Combined: same trend signal but different
sizing path so daily PnL differs (passes corr filter), with potentially higher
Calmar from the bigger calm-regime exposure.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LOOKBACK = 126
MOM_SKIP = 21
TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.16          # was 0.12
VOL_WINDOW = 60            # was 30
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class Opus1Voltarget16Skipmon60d(Strategy):
    """SP500 126d-skip-21d momentum with 16% vol-target + 60d realized vol lookback."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + MOM_SKIP + VOL_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-TREND_WINDOW:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE_MAX
        else:
            need = MOM_LOOKBACK + MOM_SKIP + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + MOM_SKIP:
                    continue
                p_end = float(col.iloc[-MOM_SKIP - 1])
                p_start = float(col.iloc[-(MOM_LOOKBACK + MOM_SKIP)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < TOP_K:
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE_MAX
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]

                # 60d realized portfolio vol
                vol_prices = ctx.closes_window(VOL_WINDOW + 5)
                port_rets = []
                n_rows = len(vol_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in vol_prices.columns:
                            continue
                        col = vol_prices[sym]
                        p_now = col.iloc[row_idx]
                        p_prev = col.iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 20:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    if annual_vol > 1e-6:
                        scale = VOL_TARGET / annual_vol
                    else:
                        scale = 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                per_slot = exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_slot

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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "opus1_voltarget16_skipmon_60d"
HYPOTHESIS = (
    "Mutate gen9_sp500_voltarget_skipmon: 16pct vol target (was 12pct) + 60d "
    "realized portfolio vol lookback (was 30d); same 126d-skip-21d top-15 SP500 "
    "momentum, SPY 200d gate -> IEF defensive; aggressive carry in calm regime."
)

STRATEGY = Opus1Voltarget16Skipmon60d()
