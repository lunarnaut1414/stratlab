"""gen_9 opus-1 — Vol-percentile carry mutation: aggressive thresholds.

Parent: gen9_gen9_sp500_vol_pct_skipmon_momentum (IS Calmar 0.65, corr 0.63,
lmc 0.55 — orthogonal-loss-mode candidate).
Mutation:
  - CALM threshold 0.75 -> 0.90 (more aggressive carry: only top decile of
    realized vol shifts to neutral)
  - STRESS threshold 0.90 -> 0.95 (only top 5% goes defensive)
  - Same 126d-skip-21d momentum, top-20 SP500, inverse-vol weighted
  - Same regime branches: calm = stocks, stress = TLT, neutral = SPY

Rationale: The parent's IS Calmar 0.65 with low corr (0.63) and low loss-mode
corr (0.55) makes it an attractive diversifier — but Calmar is weak. The parent
goes neutral (SPY only) when 25% of trailing vol-percentile windows fire, which
sacrifices a lot of stock-selection carry in benign-but-not-pristine regimes.
Raising the calm threshold to 0.90 means we stay in stocks except when vol
genuinely spikes. The lmc 0.55 ortho property is structural to the vol-pct gate,
should survive threshold tweaks. Goal: lift Calmar toward 0.8 while preserving
the low-corr / low-lmc properties.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LOOKBACK = 126
MOM_SKIP = 21
VOL_WINDOW = 20
VOL_PCT_WINDOW = 252
CALM_THRESHOLD = 0.90        # was 0.75
STRESS_THRESHOLD = 0.95      # was 0.90
TOP_K = 20
INV_VOL_WINDOW = 20
EXPOSURE = 0.97


class Opus1VolPctSkipmonAggressive(Strategy):
    """SP500 126d-skip-21d momentum with AGGRESSIVE realized-vol-percentile gates."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + MOM_SKIP + VOL_PCT_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < VOL_PCT_WINDOW + VOL_WINDOW + 5:
            return []

        spy_close = spy_hist["close"].dropna()
        spy_logret = np.log(spy_close.values[1:] / spy_close.values[:-1])

        needed = VOL_WINDOW + VOL_PCT_WINDOW
        if len(spy_logret) < needed:
            return []

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
        vol_pct = float(np.mean([v < current_vol for v in rolling_vols[1:]]))

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if vol_pct > STRESS_THRESHOLD:
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE
        elif vol_pct < CALM_THRESHOLD:
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
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum > 0:
                    for sym in ranked:
                        target[sym] = EXPOSURE * inv_vols[sym] / iv_sum
        else:
            if "SPY" in closes_now.index:
                target["SPY"] = EXPOSURE

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

NAME = "opus1_vol_pct_skipmon_aggressive"
HYPOTHESIS = (
    "Mutate gen9_sp500_vol_pct_skipmon_momentum: aggressive thresholds "
    "(calm<0.90 was 0.75, stress>0.95 was 0.90); same 126d-skip-21d top-20 SP500 "
    "inv-vol weighted; TLT stress / SPY neutral / stocks calm; lift carry while "
    "preserving low-corr (0.63) and orthogonal loss-mode (lmc 0.55)."
)

STRATEGY = Opus1VolPctSkipmonAggressive()
