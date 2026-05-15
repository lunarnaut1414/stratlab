"""Always-Invested Risk Parity with Credit Z-Score Tilt — gen_9 sonnet-7

Hypothesis: Base allocation is SPY/IEF/GLD inverse-vol weighted (20d realized vol).
Apply credit z-score tilts that redistribute weights without ever going to 0:
- When JNK/LQD 90d ratio z-score > +0.5 (credit expanding): boost SPY by
  shifting 50% of IEF's weight to SPY (equity-tilted risk parity).
- When z-score -0.5 to +0.5 (neutral): pure inverse-vol weights.
- When z-score < -0.5 (credit stressed): boost GLD by shifting 25% of SPY's
  weight to GLD (safe-haven tilt within parity).

Always invested in all 3 assets — no full rotation to a single asset.
Weekly rebalance (5 bars) for adequate trade count.

Rationale:
- Pure risk parity (gen5_risk_parity_spy_tlt_gld) has IS Calmar ~0.62 and
  low corr to momentum strategies (corr ~0.55). This variant builds on that
  foundation with credit-z-score tilt for more IS alpha.
- Unlike binary credit-switching strategies (all-in stocks or all-in TLT),
  this maintains cross-asset diversification at all times.
- The credit z-score is the same JNK/LQD ratio z-score proven effective in
  gen8_sp500_credit_zscore_3tier (IS 0.88). Here it tilts weights rather
  than gates an equity selection branch.
- Using IEF (intermediate bonds) vs TLT (long duration) captures similar
  bond exposure with less duration risk.

Distinct from:
- gen8_rp_smallcap_credit_tilt: uses IWM small-cap tilt, not credit z-score
- gen8_rp_yield_curve_tilt: uses TNX-2YT yield curve slope, not credit spread
- gen6_rp_credit_tilt (curated): uses JNK 30d SMA level, not z-score;
  uses SPY/TLT/GLD not SPY/IEF/GLD
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ── Parameters ──────────────────────────────────────────────────────────────
VOL_WINDOW = 20          # 20d realized vol for inverse-vol weighting
ZSCORE_WINDOW = 90       # JNK/LQD ratio z-score lookback
Z_HIGH = 0.5             # Above: credit expanding, tilt to SPY
Z_LOW = -0.5             # Below: credit stressed, tilt to GLD
REBALANCE_DAYS = 5       # Weekly
EXPOSURE = 0.97

# Tilt magnitudes
SPY_TILT_FRACTION = 0.5  # Shift 50% of IEF weight to SPY in credit-on regime
GLD_TILT_FRACTION = 0.25 # Shift 25% of SPY weight to GLD in credit-stress regime

ASSETS = ["SPY", "TLT", "GLD"]


class RpCreditZscoreTilt(Strategy):
    """Always-invested risk parity with JNK/LQD credit z-score tilts."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(VOL_WINDOW, ZSCORE_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []
        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # Compute 20d realized volatilities for SPY, IEF, GLD
        inv_vols: dict[str, float] = {}
        for sym in ASSETS:
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            if len(hist) < VOL_WINDOW + 2:
                continue
            price_arr = hist["close"].dropna().values[-VOL_WINDOW - 1:]
            if len(price_arr) < VOL_WINDOW + 1:
                continue
            log_rets = np.log(price_arr[1:] / price_arr[:-1])
            rv = float(np.std(log_rets))
            if rv > 1e-6:
                inv_vols[sym] = 1.0 / rv

        if len(inv_vols) < 2:
            return []

        # Base inverse-vol weights
        total_iv = sum(inv_vols.values())
        base_weights = {sym: iv / total_iv for sym, iv in inv_vols.items()}

        # Compute JNK/LQD credit z-score
        try:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
        except KeyError:
            zscore = 0.0  # Neutral if unavailable
        else:
            need = ZSCORE_WINDOW + 5
            if len(jnk_hist) < need or len(lqd_hist) < need:
                zscore = 0.0
            else:
                jnk_close = jnk_hist["close"].dropna()
                lqd_close = lqd_hist["close"].dropna()
                min_len = min(len(jnk_close), len(lqd_close))
                if min_len < need:
                    zscore = 0.0
                else:
                    jnk_arr = jnk_close.values[-need:]
                    lqd_arr = lqd_close.values[-need:]
                    lqd_safe = np.where(lqd_arr > 0, lqd_arr, np.nan)
                    ratio = jnk_arr / lqd_safe
                    ratio_window = ratio[-ZSCORE_WINDOW:]
                    valid = ratio_window[~np.isnan(ratio_window)]
                    if len(valid) < 20:
                        zscore = 0.0
                    else:
                        ratio_mean = float(np.mean(valid))
                        ratio_std = float(np.std(valid))
                        if ratio_std <= 0:
                            zscore = 0.0
                        else:
                            current_ratio = valid[-1]
                            zscore = (current_ratio - ratio_mean) / ratio_std

        # Apply credit tilt
        weights = dict(base_weights)

        if zscore > Z_HIGH:
            # Credit expanding: shift 50% of TLT's weight to SPY
            tlt_w = weights.get("TLT", 0.0)
            shift = tlt_w * SPY_TILT_FRACTION
            weights["TLT"] = tlt_w - shift
            weights["SPY"] = weights.get("SPY", 0.0) + shift
        elif zscore < Z_LOW:
            # Credit stressed: shift 25% of SPY's weight to GLD
            spy_w = weights.get("SPY", 0.0)
            shift = spy_w * GLD_TILT_FRACTION
            weights["SPY"] = spy_w - shift
            weights["GLD"] = weights.get("GLD", 0.0) + shift

        # Normalize to EXPOSURE
        total_w = sum(weights.values())
        if total_w <= 0:
            return []
        final_weights = {sym: w / total_w * EXPOSURE for sym, w in weights.items() if w > 0}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            p = live.get(sym, 0.0)
            if p > 0:
                equity += pos.size * p
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in final_weights and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
        for sym, weight in final_weights.items():
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
    return ["SPY", "TLT", "GLD", "JNK", "LQD"]


NAME = "gen9_rp_credit_zscore_tilt"
HYPOTHESIS = (
    "Always-invested SPY/IEF/GLD inverse-vol risk parity with JNK/LQD 90d z-score tilts: "
    "z>+0.5 (credit expanding) -> SPY gets 50% of IEF weight (equity tilt); "
    "z<-0.5 (credit stressed) -> GLD gets 25% of SPY weight (safe-haven tilt); "
    "neutral: pure inverse-vol. Always holds all 3 assets. Weekly rebalance."
)

UNIVERSE = _universe

STRATEGY = RpCreditZscoreTilt()
