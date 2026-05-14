"""opus-1 / gen_8 — SP500 Credit Z-Score 3-Tier (QQQ-IEF Neutral)

Mutation of gen8_sp500_credit_zscore_3tier (IS Calmar 0.88, corr 0.69 — lowest
in SP500-xsect cluster).

Parent uses JNK/LQD 90d z-score 3-tier gating:
  z > +0.5: top-15 SP500 momentum at 97% (risk-on)
  -0.5 <= z <= +0.5: IEF 60% + SPY 37% (neutral)
  z < -0.5: TLT 97% (risk-off)
plus SPY 200d bear override.

This mutation changes ONLY the neutral-tier composition: IEF 60% + SPY 37%
becomes IEF 60% + QQQ 37%. Same z-score gates, same stock selection, same
biweekly rebalance. The brief explicitly suggests "preserve corr-passing
timing by varying neutral-tier composition" — preserves the low-corr property
(0.69) which is driven by the SP500-stock-pick branch's timing, while
diversifying the neutral allocation.

Rationale: SPY-as-neutral overlaps heavily with the rest of the leaderboard
(many strategies route to SPY or SPY/TLT in defensive states). Switching to
QQQ in the neutral tier creates a small but meaningful loss-mode divergence —
QQQ outperforms SPY during low-vol calm regimes (which dominate IS) and
underperforms during late-cycle stress (which the z<-0.5 tier already routes
out of). Same SPY 200d outer bear gate stays as-is.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
ZSCORE_WINDOW = 90
Z_HIGH = 0.5
Z_LOW = -0.5
TOP_K = 15
REBALANCE_DAYS = 10
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["JNK", "LQD", "TLT", "IEF", "SPY", "QQQ"]


UNIVERSE = _universe


class SP500CreditZscoreQQQNeutral(Strategy):
    """SP500 63d momentum with JNK/LQD z-score 3-tier — QQQ neutral tier."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, ZSCORE_WINDOW) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # SPY 200d outer bear gate
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target = {"TLT": EXPOSURE}
        else:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
            if len(jnk_hist) < ZSCORE_WINDOW + 5 or len(lqd_hist) < ZSCORE_WINDOW + 5:
                return []

            jnk_close = jnk_hist["close"].tail(ZSCORE_WINDOW + 5)
            lqd_close = lqd_hist["close"].tail(ZSCORE_WINDOW + 5)
            if len(jnk_close) < ZSCORE_WINDOW or len(lqd_close) < ZSCORE_WINDOW:
                return []

            min_len = min(len(jnk_close), len(lqd_close))
            jnk_vals = jnk_close.values[-min_len:]
            lqd_vals = lqd_close.values[-min_len:]
            lqd_safe = np.where(lqd_vals > 0, lqd_vals, np.nan)
            ratio = jnk_vals / lqd_safe
            ratio_window = ratio[-ZSCORE_WINDOW:]
            valid = ratio_window[~np.isnan(ratio_window)]
            if len(valid) < 20:
                return []

            ratio_mean = float(np.mean(valid))
            ratio_std = float(np.std(valid))
            if ratio_std <= 0 or not np.isfinite(ratio_std):
                return []

            current_ratio = valid[-1]
            z_score = (current_ratio - ratio_mean) / ratio_std

            if z_score < Z_LOW:
                target = {"TLT": EXPOSURE}
            elif z_score <= Z_HIGH:
                # *** MUTATION: QQQ-IEF instead of SPY-IEF neutral ***
                target = {"IEF": 0.60, "QQQ": 0.37}
            else:
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"IEF": EXPOSURE}
                else:
                    live = {s: float(closes[s]) for s in closes.index
                            if closes[s] > 0 and s not in ("JNK", "LQD", "TLT", "IEF", "SPY", "QQQ")}
                    scores: dict[str, float] = {}
                    for sym in live:
                        if sym not in prices_window.columns:
                            continue
                        col = prices_window[sym].dropna()
                        if len(col) < MOMENTUM_WINDOW:
                            continue
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-MOMENTUM_WINDOW])
                        if p_start <= 0:
                            continue
                        r = p_end / p_start - 1.0
                        if np.isfinite(r):
                            scores[sym] = r

                    if len(scores) < TOP_K:
                        target = {"IEF": EXPOSURE}
                    else:
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        selected = []
                        for sym, _ in ranked:
                            if len(selected) >= TOP_K:
                                break
                            hist = ctx.history(sym)
                            if len(hist) < TREND_WINDOW:
                                continue
                            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
                            price = live.get(sym, 0.0)
                            if price > sma:
                                selected.append(sym)
                        if not selected:
                            target = {"IEF": EXPOSURE}
                        else:
                            target = {sym: EXPOSURE / len(selected) for sym in selected}

        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


NAME = "opus1_sp500_credit_zscore_qqq_neutral"
HYPOTHESIS = (
    "Mutation of sp500_credit_zscore_3tier: same JNK/LQD 90d z-score 3-tier gating and "
    "top-15 SP500 momentum; change neutral-tier composition from IEF60+SPY37 to IEF60+QQQ37 "
    "to preserve corr-passing timing while diversifying loss-mode profile; biweekly rebalance"
)

STRATEGY = SP500CreditZscoreQQQNeutral()
