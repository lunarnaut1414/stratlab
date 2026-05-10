"""Equal-weight ensemble of 3 low-correlation gen_5 diversifiers.

Components (each sized at 1/3 of capital, target weights summed and netted):

  A. bond_equity_regime
     - Risk-on (TLT/SPY ratio < 50d MA): hold top-10 SP500 stocks by 63d
       momentum, equally weighted.
     - Risk-off (TLT/SPY ratio > 50d MA): hold TLT 60% + GLD 40%.
     - Rebalance every 10 bars.
     - Diversifier rank: corr_to_top5 = 0.00 (best in leaderboard).

  B. credit_spread_hyg_lqd
     - JNK when 20d MA of JNK/LQD ratio > 90d MA (tightening spreads, risk-on).
     - LQD when 20d MA < 90d MA (widening spreads, defensive).
     - Rebalance every 5 bars (weekly).
     - Diversifier rank: corr_to_top5 = 0.47, but uses entirely different
       asset class (credit) than the other two.

  C. halloween_sell_in_may
     - SPY in Nov-Apr (winter), TLT in May-Oct (summer).
     - Daily check (state changes only at month boundaries).
     - Diversifier rank: corr_to_top5 = 0.40 with pure-calendar signal that
       is by construction uncorrelated with any market-state regime.

Composition rule: equal-weight (1/N).
  - Each component independently produces a target weight per symbol
    (capped at COMPONENT_WEIGHT = 1/3).
  - Symbols held by multiple components net naturally (e.g. if A wants
    TLT 0.6 * 1/3 = 0.20 and C wants TLT 0.95 * 1/3 = 0.317 in summer,
    total target TLT weight ~ 0.52).
  - Final exposure capped by EXPOSURE_CAP = 0.97 to leave a tiny cash
    buffer for slippage/rounding (otherwise sums >1.0 would force margin).

Why this triplet:
  1. bond_equity_regime fires off TLT/SPY ratio (risk-appetite proxy).
  2. credit_spread_hyg_lqd fires off JNK/LQD ratio (credit-spread proxy).
  3. halloween fires off calendar month (no market data at all).
  These three signals are structurally orthogonal — none uses a signal
  that the others use, so daily-return correlations should be low.

Hypothesis: combining 3 uncorrelated low-Calmar but positive-edge
strategies should yield a higher Calmar than any one alone, because
drawdowns of one component are partially offset by the others.

IS window: 2010-01-01 to 2018-12-31. Tested via standard submit harness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — bond_equity_regime parameters
# ---------------------------------------------------------------------------
A_REBALANCE_EVERY = 10
A_MOMENTUM_WINDOW = 63
A_RATIO_MA_WINDOW = 50
A_TOP_K = 10

# ---------------------------------------------------------------------------
# Component B — credit_spread_hyg_lqd parameters
# ---------------------------------------------------------------------------
B_FAST_MA = 20
B_SLOW_MA = 90
B_REBALANCE = 5

# ---------------------------------------------------------------------------
# Component C — halloween parameters
# ---------------------------------------------------------------------------
C_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
# Summer months are everything else (5..10)

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0   # each component sized to 1/3 of capital
EXPOSURE_CAP = 0.97             # total invested cap (leave cash buffer)
ENSEMBLE_REBALANCE_EVERY = 5    # check / rebalance ensemble weekly
WARMUP_BARS = 100               # leave room for the slowest component (90d MA)
MIN_TRADE_DELTA = 1             # min absolute share delta to avoid micro-orders


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """Bond-equity regime target weights (NOT scaled by COMPONENT_WEIGHT yet)."""
    warmup = max(A_MOMENTUM_WINDOW, A_RATIO_MA_WINDOW) + 10
    if ctx.idx < warmup:
        return None

    try:
        tlt_hist = ctx.history("TLT")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None

    if len(tlt_hist) < A_RATIO_MA_WINDOW + 5 or len(spy_hist) < A_RATIO_MA_WINDOW + 5:
        return None

    tlt_close = tlt_hist["close"].iloc[-(A_RATIO_MA_WINDOW + 5):]
    spy_close = spy_hist["close"].iloc[-(A_RATIO_MA_WINDOW + 5):]
    ratio = (tlt_close / spy_close).dropna()
    if len(ratio) < A_RATIO_MA_WINDOW:
        return None

    ratio_ma = float(ratio.rolling(A_RATIO_MA_WINDOW).mean().iloc[-1])
    current_ratio = float(ratio.iloc[-1])
    if not (np.isfinite(ratio_ma) and np.isfinite(current_ratio)):
        return None

    risk_off = current_ratio > ratio_ma

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    weights: dict[str, float] = {}
    if risk_off:
        # Defensive: TLT 60%, GLD 40%
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.6
        if "GLD" in closes_now.index:
            weights["GLD"] = 0.4
    else:
        # Risk-on: top-K sp500 momentum
        prices = ctx.closes_window(A_MOMENTUM_WINDOW + 5)
        if len(prices) < A_MOMENTUM_WINDOW:
            return None

        scores: dict[str, float] = {}
        for sym in prices.columns:
            # Skip ETFs that are not part of cross-sectional momentum universe
            if sym in {"TLT", "GLD", "SPY", "JNK", "LQD"}:
                continue
            col = prices[sym].dropna()
            if len(col) < A_MOMENTUM_WINDOW:
                continue
            ret = float(col.iloc[-1] / col.iloc[-A_MOMENTUM_WINDOW] - 1.0)
            if np.isfinite(ret):
                scores[sym] = ret

        if len(scores) < A_TOP_K:
            return None

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        longs = ranked[:A_TOP_K]
        per_weight = 1.0 / len(longs)
        for sym in longs:
            weights[sym] = per_weight

    return weights


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """Credit spread (JNK/LQD) target weights (full 1.0 in chosen leg)."""
    if ctx.idx < B_SLOW_MA + 5:
        return None

    try:
        jnk_hist = ctx.history("JNK")
        lqd_hist = ctx.history("LQD")
    except KeyError:
        return None

    if len(jnk_hist) < B_SLOW_MA + 1 or len(lqd_hist) < B_SLOW_MA + 1:
        return None

    jnk_close = jnk_hist["close"].iloc[-(B_SLOW_MA + 5):]
    lqd_close = lqd_hist["close"].iloc[-(B_SLOW_MA + 5):]
    min_len = min(len(jnk_close), len(lqd_close))
    if min_len < B_SLOW_MA:
        return None

    jnk_c = jnk_close.iloc[-min_len:].values
    lqd_c = lqd_close.iloc[-min_len:].values
    ratio = jnk_c / lqd_c

    fast_val = float(np.mean(ratio[-B_FAST_MA:]))
    slow_val = float(np.mean(ratio[-B_SLOW_MA:]))
    if not (np.isfinite(fast_val) and np.isfinite(slow_val)):
        return None

    closes_now = ctx.closes()
    target_sym = "JNK" if fast_val > slow_val else "LQD"
    if target_sym not in closes_now.index:
        return None

    return {target_sym: 1.0}


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Halloween seasonal target weights (full 1.0 in chosen leg)."""
    if ctx.idx < 5:
        return None

    closes_now = ctx.closes()
    if "SPY" not in closes_now.index or "TLT" not in closes_now.index:
        return None

    month = ctx.timestamp.month
    if month in C_WINTER_MONTHS:
        return {"SPY": 1.0}
    return {"TLT": 1.0}


class EnsembleBondCreditSeasonal(Strategy):
    """Equal-weight ensemble of 3 low-correlation gen_5 diversifiers.

    Each on_bar call recomputes target weights from each component
    independently, sums them with weight COMPONENT_WEIGHT = 1/3, caps
    total exposure at 0.97, and emits delta orders to reach target.
    """

    def __init__(
        self,
        rebalance_every: int = ENSEMBLE_REBALANCE_EVERY,
        component_weight: float = COMPONENT_WEIGHT,
        exposure_cap: float = EXPOSURE_CAP,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            component_weight=component_weight,
            exposure_cap=exposure_cap,
        )
        self.rebalance_every = rebalance_every
        self.component_weight = component_weight
        self.exposure_cap = exposure_cap

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Compute each component's target weights (or skip if unavailable)
        a_w = _component_a_weights(ctx)
        b_w = _component_b_weights(ctx)
        c_w = _component_c_weights(ctx)

        # Aggregate weighted targets across all components.
        # Each component contributes its raw weights * component_weight.
        # Components currently unavailable (None) contribute 0; the
        # ensemble degrades gracefully to remaining components.
        target: dict[str, float] = {}
        for component_targets in (a_w, b_w, c_w):
            if component_targets is None:
                continue
            for sym, w in component_targets.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

        # Apply exposure cap by scaling down if total > cap
        total_w = sum(target.values())
        if total_w > self.exposure_cap and total_w > 0.0:
            scale = self.exposure_cap / total_w
            target = {sym: w * scale for sym, w in target.items()}

        # If no component produced anything, do nothing (stay flat / hold cash)
        if not target:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        # Build target share counts
        target_shares: dict[str, int] = {}
        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            shares = int(portfolio_value * weight / price)
            if shares > 0:
                target_shares[sym] = shares

        orders: list[Order] = []

        # Sells first to free up cash before buys
        # 1. Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # 2. Reduce positions overweight vs target
        for sym, tgt in target_shares.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta < -MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        # 3. Buys to reach target
        for sym, tgt in target_shares.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta > MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


def _universe() -> list[str]:
    """Universe = SP500 (for component A momentum) plus the ETFs each
    component routes to."""
    from stratlab.data.universe import sp500_tickers
    extra = ["SPY", "TLT", "GLD", "JNK", "LQD"]
    return sp500_tickers() + extra


NAME = "ensemble_bond_credit_seasonal"
HYPOTHESIS = (
    "Equal-weight ensemble of 3 low-correlation diversifiers: bond_equity_regime "
    "(TLT/SPY ratio gating SP500 momentum/defensive), credit_spread_hyg_lqd "
    "(JNK/LQD spread MA crossover), and halloween (SPY winter / TLT summer); "
    "each component sized at 1/3 capital, overlapping positions netted, "
    "total exposure capped at 0.97."
)
UNIVERSE = _universe

STRATEGY = EnsembleBondCreditSeasonal()
