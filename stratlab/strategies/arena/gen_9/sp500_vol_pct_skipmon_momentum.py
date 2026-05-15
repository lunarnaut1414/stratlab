"""SP500 Skip-Month Momentum gated by SPY realized-vol percentile rank.

Hypothesis (sonnet-5, gen_9):
    SP500 cross-sectional momentum gated by SPY 20d realized-vol 90d
    percentile rank:
    - vol-pct < 40th pct (calm regime)  -> top-20 SP500 by 126d-skip-21d
      momentum, inverse-vol weighted
    - vol-pct > 70th pct (stress)       -> TLT 97%
    - neutral (40-70th pct)             -> SPY 97%
    Biweekly rebalance. Gate = percentile rank of rolling vol, not absolute
    VIX level or MA crossover.

Diversification angle vs leaderboard:
  - gen8_sp500_skipmon_63sma_momentum (OOS Calmar 0.63): uses SPY 63d SMA
    trend gate + individual stock 50d SMA — this strategy replaces BOTH gates
    with a single realized-vol percentile gate. Daily PnL path differs because
    SPY vol-pct fires differently than price-trend crossovers.
  - Existing VIX gates use absolute levels (VIX<25, VIX<20). Percentile-rank
    is robust to regime shifts in vol levels (2010 vs 2017 VIX baselines differ
    dramatically).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # 6 months
MOM_SKIP = 21               # skip last 1 month (Jegadeesh-Titman)
VOL_WINDOW = 20             # realized vol window (days)
VOL_PCT_WINDOW = 252        # lookback for percentile rank of vol (1 year)
CALM_THRESHOLD = 0.75       # vol pct below this -> stock selection
STRESS_THRESHOLD = 0.90     # vol pct above this -> defensive
TOP_K = 20
INV_VOL_WINDOW = 20
EXPOSURE = 0.97


class Sp500VolPctSkipmonMomentum(Strategy):
    """SP500 126d-skip-21d momentum with realized-vol-percentile regime gate."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + MOM_SKIP + VOL_PCT_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # --- Compute SPY 20d realized-vol percentile over trailing 90d ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < VOL_PCT_WINDOW + VOL_WINDOW + 5:
            return []

        spy_close = spy_hist["close"].dropna()
        spy_logret = np.log(spy_close.values[1:] / spy_close.values[:-1])

        # Need enough data: most recent 20d vol + 90d of trailing vols
        needed = VOL_WINDOW + VOL_PCT_WINDOW
        if len(spy_logret) < needed:
            return []

        # Rolling 20d vol over trailing 90d windows
        rolling_vols = []
        for i in range(VOL_PCT_WINDOW):
            end = len(spy_logret) - i
            start = end - VOL_WINDOW
            if start < 0:
                break
            rolling_vols.append(float(np.std(spy_logret[start:end])))

        if len(rolling_vols) < 10:
            return []

        current_vol = rolling_vols[0]
        # Percentile rank: what fraction of trailing 90d 20d-vols < current_vol
        vol_pct = float(np.mean([v < current_vol for v in rolling_vols[1:]]))

        # --- Determine regime ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if vol_pct > STRESS_THRESHOLD:
            # Stress — defensive TLT
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE
        elif vol_pct < CALM_THRESHOLD:
            # Calm — SP500 cross-sectional momentum
            need = MOM_LOOKBACK + MOM_SKIP + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + MOM_SKIP:
                    continue
                p_end = float(col.iloc[-MOM_SKIP - 1])
                p_start = float(col.iloc[-(MOM_LOOKBACK + MOM_SKIP)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                tail = col.iloc[-INV_VOL_WINDOW - 1:]
                if len(tail) < INV_VOL_WINDOW + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < TOP_K:
                # Fall back to SPY when not enough candidates
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum > 0:
                    for sym in ranked:
                        target[sym] = EXPOSURE * inv_vols[sym] / iv_sum
        else:
            # Neutral — SPY
            if "SPY" in closes_now.index:
                target["SPY"] = EXPOSURE

        # --- Generate orders ---
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
    return sp500_tickers() + ["SPY", "TLT"]


UNIVERSE = _universe

NAME = "gen9_sp500_vol_pct_skipmon_momentum"
HYPOTHESIS = (
    "SP500 cross-sectional momentum gated by SPY 20d realized-vol 90d percentile rank: "
    "calm (<40th pct) -> top-20 SP500 by 126d-skip-21d momentum inverse-vol weighted; "
    "stress (>70th pct) -> TLT 97%; neutral -> SPY 97%; biweekly rebalance."
)

STRATEGY = Sp500VolPctSkipmonMomentum()
