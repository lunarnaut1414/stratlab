"""SP500 Credit Z-Score 3-Tier Gated Momentum — gen_8 sonnet-9

Hypothesis: Use a rolling z-score of the JNK/LQD credit spread ratio as
the regime gate for SP500 momentum. The z-score is normalized against a
90-day rolling window, which makes the signal self-adjusting to recent
spread levels (unlike simple MA crossover which has slow-moving reference).

Three tiers:
1. z > +0.5 (credit tightening significantly): hold top-15 SP500 by 63d
   momentum above 200d SMA at 97% exposure.
2. -0.5 <= z <= +0.5 (neutral credit): hold IEF 60% + SPY 37% (no stock
   selection risk, but not fully defensive).
3. z < -0.5 (credit widening significantly): hold TLT 97% (full defensive).

SPY 200d SMA bear override: if SPY below 200d SMA, force TLT regardless.
Biweekly rebalance (every 10 bars).

Rationale: Credit spread z-score is a more sensitive signal than MA crossover
because it normalizes for the recent spread regime. A small MA crossover
during a generally wide-spread environment should be treated differently from
the same crossover during a tight-spread environment. The z-score captures
this. The 3-tier allocation provides a graceful degradation into neutral (IEF)
rather than binary risk-on/risk-off.

Differentiation: gen5_opus1_credit_zscore_breakout (IS Calmar 0.61, corr 0.55)
uses 21-day skip-1 momentum on top-10 stocks and no SPY 200d outer gate. This
strategy uses 63d momentum on top-15 with SPY 200d outer gate and IEF in the
neutral tier — different stock selection timing and different neutral allocation.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month momentum for stock ranking
TREND_WINDOW = 200        # SPY 200d SMA bear gate
ZSCORE_WINDOW = 90        # Rolling window for credit spread z-score
Z_HIGH = 0.5              # Above this: risk-on (stocks)
Z_LOW = -0.5              # Below this: risk-off (TLT)
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["JNK", "LQD", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class Sp500CreditZscore3Tier(Strategy):
    """SP500 63d momentum with JNK/LQD z-score 3-tier regime gate."""

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

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target = {"TLT": EXPOSURE}
        else:
            # --- Compute JNK/LQD ratio z-score ---
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")

            if len(jnk_hist) < ZSCORE_WINDOW + 5 or len(lqd_hist) < ZSCORE_WINDOW + 5:
                return []

            # Align on common dates and compute ratio
            jnk_close = jnk_hist["close"].tail(ZSCORE_WINDOW + 5)
            lqd_close = lqd_hist["close"].tail(ZSCORE_WINDOW + 5)

            if len(jnk_close) < ZSCORE_WINDOW or len(lqd_close) < ZSCORE_WINDOW:
                return []

            min_len = min(len(jnk_close), len(lqd_close))
            jnk_vals = jnk_close.values[-min_len:]
            lqd_vals = lqd_close.values[-min_len:]

            # JNK/LQD ratio
            lqd_safe = np.where(lqd_vals > 0, lqd_vals, np.nan)
            ratio = jnk_vals / lqd_safe

            # Use up to ZSCORE_WINDOW values
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

            # --- Route based on z-score ---
            if z_score < Z_LOW:
                # Credit widening: TLT defensive
                target = {"TLT": EXPOSURE}
            elif z_score <= Z_HIGH:
                # Neutral credit: IEF + SPY blend
                target = {"IEF": 0.60, "SPY": 0.37}
            else:
                # Credit tightening: SP500 momentum
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"IEF": EXPOSURE}
                else:
                    live = {s: float(closes[s]) for s in closes.index
                            if closes[s] > 0 and s not in ("JNK", "LQD", "TLT", "IEF", "SPY")}

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

                        # Apply 200d SMA filter on candidates
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

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
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


NAME = "sp500_credit_zscore_3tier"
HYPOTHESIS = (
    "SP500 top-15 by 63d momentum with JNK/LQD credit-spread 90d z-score gate: "
    "z>+0.5 hold top-15 SP500 stocks 97%; z-0.5 to +0.5 hold IEF+SPY 60/37; "
    "z<-0.5 hold TLT 97%; SPY 200d bear override to TLT; biweekly rebalance."
)

STRATEGY = Sp500CreditZscore3Tier()
