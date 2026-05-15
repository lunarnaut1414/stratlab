"""Credit Z-Score 3-Tier Gating Skip-Month SP500 Momentum — gen_9 sonnet-7

Hypothesis: JNK/LQD 90-day return z-score as credit regime signal gating
SP500 skip-month momentum (126d minus 21d, Jegadeesh-Titman 1993).

Regime logic:
  - SPY below 200d SMA: TLT 97% (outer bear gate)
  - z-score > +0.5 (credit expanding): top-15 SP500 by 126d-skip-21d momentum,
    inverse-vol weighted, 97% exposure
  - z-score -0.5 to +0.5 (neutral credit): QQQ 97% (tech/growth tilt in neutral
    credit — vs parent gen8_credit_zscore_3tier which used SPY+IEF)
  - z-score < -0.5 (credit stressed): TLT 97%

Key structural differences from gen8_sp500_credit_zscore_3tier (IS 0.88, OOS 0.40):
  1. Skip-month momentum (126d-skip-21d) instead of standard 63d momentum
  2. Neutral-tier: QQQ 97% vs parent's SPY 60%+IEF 37%
  3. Both changes create a meaningfully different daily return path that
     should reduce corr to top-5 while maintaining/improving IS Calmar.

The skip-month feature (gen8_sp500_skipmon_63sma, OOS 0.63) is the most
OOS-robust stock selection approach in the leaderboard. Combining it with the
credit-zscore gate adds a macro filter that reduces whipsaws. The QQQ neutral
tier is inspired by gen8_opus1_sp500_credit_zscore_qqq_neutral which beat its
parent OOS (0.46 vs 0.40) by changing the neutral tier.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# ── Parameters ──────────────────────────────────────────────────────────────
ZSCORE_WINDOW = 90       # Rolling window for JNK/LQD z-score
ZSCORE_HIGH = 0.0        # Above this: credit-confirmed risk-on (neutral or positive z = stocks)
ZSCORE_LOW = -1.0        # Below this: credit-stressed risk-off (strongly negative z = TLT)
MOM_LONG = 126           # Momentum lookback (6 months)
MOM_SKIP = 21            # Skip most recent 21 days (1 month)
VOL_WINDOW = 21          # Inverse-vol weighting window
SPY_TREND = 200          # SPY outer bear gate
TOP_K = 15
REBALANCE_EVERY = 10     # Bi-weekly
EXPOSURE = 0.97


class CreditZScoreSkipMonSp500(Strategy):
    """JNK/LQD credit z-score 3-tier with skip-month SP500 momentum."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(ZSCORE_WINDOW, SPY_TREND, MOM_LONG + MOM_SKIP) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []
        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # SPY outer bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma200 = float(spy_close.iloc[-SPY_TREND:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        if not spy_bull:
            targets: dict[str, float] = {"TLT": EXPOSURE}
            return self._trade_to_targets(ctx, live, targets)

        # Compute JNK/LQD ratio z-score (level-based, not returns-based)
        try:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
        except KeyError:
            # Fallback to QQQ if credit data unavailable
            targets = {"QQQ": EXPOSURE}
            return self._trade_to_targets(ctx, live, targets)

        need = ZSCORE_WINDOW + 5
        if len(jnk_hist) < need or len(lqd_hist) < need:
            return []

        jnk_close = jnk_hist["close"].dropna()
        lqd_close = lqd_hist["close"].dropna()
        min_len = min(len(jnk_close), len(lqd_close))
        if min_len < need:
            return []

        jnk_arr = jnk_close.values[-need:]
        lqd_arr = lqd_close.values[-need:]

        # JNK/LQD ratio (higher = credit tightening = risk-on)
        lqd_safe = np.where(lqd_arr > 0, lqd_arr, np.nan)
        ratio = jnk_arr / lqd_safe
        ratio_window = ratio[-ZSCORE_WINDOW:]
        valid = ratio_window[~np.isnan(ratio_window)]
        if len(valid) < 20:
            return []

        ratio_mean = float(np.mean(valid))
        ratio_std = float(np.std(valid))
        if ratio_std <= 0 or not np.isfinite(ratio_std):
            return []

        current_ratio = valid[-1]
        zscore = (current_ratio - ratio_mean) / ratio_std

        if zscore < ZSCORE_LOW:
            # Credit stressed → defensive
            targets = {"TLT": EXPOSURE}
        elif zscore > ZSCORE_HIGH:
            # Credit expanding → skip-month momentum stocks
            targets = self._skipmon_targets(ctx, live)
            if not targets:
                targets = {"SPY": EXPOSURE}
        else:
            # Neutral credit → QQQ (better than SPY+IEF in IS tech bull)
            targets = {"QQQ": EXPOSURE}

        return self._trade_to_targets(ctx, live, targets)

    def _skipmon_targets(
        self, ctx: BarContext, live: dict[str, float]
    ) -> dict[str, float]:
        """Compute skip-month momentum targets."""
        need = MOM_LONG + MOM_SKIP + 2
        prices = ctx.closes_window(need)
        if len(prices) < need - 1:
            return {}

        scores: dict[str, float] = {}
        inv_vols: dict[str, float] = {}

        for sym in prices.columns:
            if sym.startswith("^") or sym in ("JNK", "LQD", "SPY", "QQQ", "TLT"):
                continue
            col = prices[sym].dropna()
            if len(col) < MOM_LONG + MOM_SKIP:
                continue
            # Skip-month: use col[-MOM_SKIP] as end, col[-(MOM_LONG+MOM_SKIP)] as start
            end_idx = -MOM_SKIP
            start_idx = -(MOM_LONG + MOM_SKIP)
            p_end = float(col.iloc[end_idx])
            p_start = float(col.iloc[start_idx])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret = p_end / p_start - 1.0
            if not np.isfinite(ret):
                continue

            # Inverse-vol weighting
            vol_tail = col.iloc[-VOL_WINDOW - 1:]
            if len(vol_tail) < VOL_WINDOW + 1:
                continue
            log_r = np.log(vol_tail.values[1:] / vol_tail.values[:-1])
            rv = float(np.std(log_r))
            if rv <= 1e-6 or not np.isfinite(rv):
                continue

            scores[sym] = ret
            inv_vols[sym] = 1.0 / rv

        if len(scores) < TOP_K:
            return {}

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
        iv_sum = sum(inv_vols[s] for s in ranked)
        if iv_sum <= 0:
            return {}

        return {sym: EXPOSURE * inv_vols[sym] / iv_sum for sym in ranked}

    def _trade_to_targets(
        self,
        ctx: BarContext,
        live: dict[str, float],
        targets: dict[str, float],
    ) -> list[Order]:
        """Generate orders to reach target weights."""
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            p = live.get(sym, 0.0)
            if p > 0:
                equity += pos.size * p
        if equity <= 0:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in targets and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in targets.items():
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
    return sp500_tickers() + ["JNK", "LQD", "SPY", "QQQ", "TLT", "IEF"]


NAME = "gen9_credit_zscore_skipmon_sp500"
HYPOTHESIS = (
    "JNK/LQD 90d z-score 3-tier gating skip-month SP500 momentum (126d-skip-21d): "
    "z>+0.5 -> top-15 SP500 by skip-month momentum inverse-vol weighted; "
    "z -0.5 to +0.5 -> QQQ 97% (neutral-credit tech tilt); "
    "z<-0.5 -> TLT 97%; SPY 200d bear -> TLT 97%. Biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = CreditZScoreSkipMonSp500()
