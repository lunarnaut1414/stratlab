"""Ensemble of 3 orthogonal strategies: quality momentum + bond term structure + seasonal.

Hypothesis: Equal-weight ensemble of:
  A. NearHi Quality Momentum (nearhi_momentum_quality) — SP500 stocks near 52w high
     with strong 126d momentum, inverse-vol weighted, SPY 200d gate (1/3 capital)
  B. Bond Term Structure Rotation — TLT/IEF/SHY allocated by 10Y-3M yield curve
     slope via ^TNX-^IRX signal (1/3 capital)
  C. Halloween / Sell-in-May Seasonal — SPY Nov-Apr, TLT May-Oct (1/3 capital)

Rationale:
  - Component A (quality equity momentum): high IS Calmar 1.16 in leaderboard,
    picks SP500 stocks near 52w high with strong momentum. Orthogonal to bond
    strategies and seasonal strategies.
  - Component B (bond term structure): IS Calmar 1.12 in leaderboard, pure rates
    signal that is orthogonal to equity selection and calendar effects.
  - Component C (halloween seasonal): IS Calmar 0.51, calendar-driven with 0.40
    corr to top-5, structurally orthogonal to both market-state regimes.
  - Key insight: these 3 components use completely different signal sources
    (equity price/quality, treasury yield curve, calendar month). Their drawdowns
    should occur at different times, providing genuine diversification.
  - The ensemble of the best bond strategy + the best equity quality strategy +
    the most orthogonal seasonal creates a portfolio that should have:
    * Lower max drawdown than any individual component
    * IS Calmar well above 0.5 due to high-Calmar components

Distinct from existing ensembles:
  - gen5_ensemble_bond_credit_seasonal uses bond_equity_regime + credit_spread + halloween
    (3 market-state signals). This replaces bond_equity_regime with nearhi quality
    stock selection (higher IS Calmar) and credit_spread with bond term structure
    (truly different — rates-based not credit-based).
  - gen6_opus3_ensemble_credit_sector_breadth uses hy_credit + sector + smallcap_leadership
    (all market-state signals). This uses calendar seasonality + rates + quality stock
    selection (3 fundamentally different signal types).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — NearHi Quality Momentum parameters
# ---------------------------------------------------------------------------
A_REBALANCE_EVERY = 21     # monthly
A_MOMENTUM_WINDOW = 126    # 6-month
A_HIGH_WINDOW = 252        # 52-week
A_NEARHI_THRESHOLD = 0.80  # price > 80% of 52w high
A_VOL_WINDOW = 20          # for inverse-vol weights
A_TOP_K = 15
A_TREND_WINDOW = 200       # SPY 200d SMA gate

# ---------------------------------------------------------------------------
# Component B — Bond Term Structure parameters
# ---------------------------------------------------------------------------
B_STEEP_THRESHOLD = 2.5    # very steep (%)
B_MOD_STEEP_THRESHOLD = 1.5  # moderately steep (%)
B_FLAT_THRESHOLD = 0.5     # very flat (%)
B_SMOOTH_DAYS = 20         # smooth the curve slope signal
B_REBALANCE_EVERY = 10     # biweekly

# ---------------------------------------------------------------------------
# Component C — Halloween seasonal
# ---------------------------------------------------------------------------
C_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
# Summer months = {5, 6, 7, 8, 9, 10}

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97
ENSEMBLE_REBALANCE_EVERY = 5  # weekly ensemble check

WARMUP_BARS = A_HIGH_WINDOW + 10  # slowest component drives warmup


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return (
        sp500_tickers()
        + ["SPY", "TLT", "IEF", "SHY", "GLD", "^TNX", "^IRX"]
    )


UNIVERSE = _universe


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """NearHi Quality Momentum target weights (un-scaled, sum to ~1.0)."""
    if ctx.idx < A_HIGH_WINDOW + 10:
        return None
    if ctx.idx % A_REBALANCE_EVERY != 0:
        # Only rebalance on scheduled bars; otherwise hold current weights
        # Return None to signal "don't rebalance this component this bar"
        return None

    # SPY 200d SMA gate
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < A_TREND_WINDOW + 5:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < A_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-A_TREND_WINDOW:].mean())
    bull = float(spy_close.iloc[-1]) > spy_sma

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    if not bull:
        if "TLT" in closes_now.index:
            return {"TLT": 1.0}
        return None

    # Near-52w-high quality filter + 126d momentum + inverse-vol
    need = A_HIGH_WINDOW + 5
    prices = ctx.closes_window(need)
    if len(prices) < need - 5:
        return None

    scores: dict[str, float] = {}
    inv_vols: dict[str, float] = {}

    for sym in prices.columns:
        if sym in {"TLT", "IEF", "SHY", "SPY", "GLD"}:
            continue
        col = prices[sym].dropna()
        if len(col) < A_HIGH_WINDOW:
            continue

        # Near-52w-high quality filter
        w52_high = float(col.iloc[-A_HIGH_WINDOW:].max())
        if w52_high <= 0 or not np.isfinite(w52_high):
            continue
        current_price = float(col.iloc[-1])
        nearhi_ratio = current_price / w52_high
        if nearhi_ratio < A_NEARHI_THRESHOLD:
            continue

        # 126d momentum
        if len(col) < A_MOMENTUM_WINDOW + 2:
            continue
        p_end = float(col.iloc[-1])
        p_start = float(col.iloc[-A_MOMENTUM_WINDOW])
        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
            continue
        ret = p_end / p_start - 1.0
        if not np.isfinite(ret):
            continue

        # Inverse-vol
        tail = col.iloc[-A_VOL_WINDOW - 1:]
        if len(tail) < A_VOL_WINDOW + 1:
            continue
        logr = np.log(tail.values[1:] / tail.values[:-1])
        rv = float(np.std(logr))
        if rv <= 1e-6 or not np.isfinite(rv):
            continue

        scores[sym] = ret
        inv_vols[sym] = 1.0 / rv

    if len(scores) < 5:
        if "TLT" in closes_now.index:
            return {"TLT": 1.0}
        return None

    k = min(A_TOP_K, len(scores))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    iv_sum = sum(inv_vols[s] for s in ranked)
    if iv_sum <= 0:
        return None

    weights: dict[str, float] = {}
    for sym in ranked:
        weights[sym] = inv_vols[sym] / iv_sum
    return weights


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """Bond Term Structure target weights (un-scaled, sum to ~1.0)."""
    warmup = max(B_SMOOTH_DAYS + 10, 30)
    if ctx.idx < warmup:
        return None
    if ctx.idx % B_REBALANCE_EVERY != 0:
        return None

    # Compute smoothed curve slope ^TNX - ^IRX
    slope = float("nan")
    try:
        tnx_hist = ctx.history("^TNX")
        irx_hist = ctx.history("^IRX")
        if (
            tnx_hist is not None
            and irx_hist is not None
            and len(tnx_hist) >= B_SMOOTH_DAYS
            and len(irx_hist) >= B_SMOOTH_DAYS
        ):
            tnx_close = tnx_hist["close"].dropna()
            irx_close = irx_hist["close"].dropna()
            if len(tnx_close) >= B_SMOOTH_DAYS and len(irx_close) >= B_SMOOTH_DAYS:
                df = pd.concat(
                    [tnx_close.rename("tnx"), irx_close.rename("irx")],
                    axis=1,
                ).dropna()
                if len(df) >= B_SMOOTH_DAYS:
                    slopes = df["tnx"] - df["irx"]
                    slope = float(slopes.iloc[-B_SMOOTH_DAYS:].mean())
    except Exception:
        pass

    if not np.isfinite(slope):
        slope = 1.5  # safe neutral

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    if slope >= B_STEEP_THRESHOLD:
        weights = {}
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.65
        if "IEF" in closes_now.index:
            weights["IEF"] = 0.35
        return weights if weights else None
    elif slope >= B_MOD_STEEP_THRESHOLD:
        weights = {}
        if "IEF" in closes_now.index:
            weights["IEF"] = 0.65
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.35
        return weights if weights else None
    elif slope >= B_FLAT_THRESHOLD:
        if "IEF" in closes_now.index:
            return {"IEF": 1.0}
        return None
    else:
        if "SHY" in closes_now.index:
            return {"SHY": 1.0}
        return None


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Halloween seasonal weights (full 1.0 in chosen leg)."""
    if ctx.idx < 5:
        return None
    closes_now = ctx.closes()
    if "SPY" not in closes_now.index or "TLT" not in closes_now.index:
        return None
    month = ctx.timestamp.month
    if month in C_WINTER_MONTHS:
        return {"SPY": 1.0}
    return {"TLT": 1.0}


class EnsembleQualityTermstructSeasonal(Strategy):
    """Equal-weight ensemble of nearhi quality momentum + bond termstruct + halloween."""

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
        self.rebalance_every = int(rebalance_every)
        self.component_weight = float(component_weight)
        self.exposure_cap = float(exposure_cap)
        # Track last-computed component targets for components that don't rebalance every bar
        self._last_a: dict[str, float] = {}
        self._last_b: dict[str, float] = {}
        self._last_c: dict[str, float] = {}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get component weights (update cache if component fires this bar)
        a = _component_a_weights(ctx)
        if a is not None:
            self._last_a = a

        b = _component_b_weights(ctx)
        if b is not None:
            self._last_b = b

        c = _component_c_weights(ctx)
        if c is not None:
            self._last_c = c

        # Aggregate: sum component weights * COMPONENT_WEIGHT
        combined: dict[str, float] = {}
        for sym, w in self._last_a.items():
            combined[sym] = combined.get(sym, 0.0) + w * self.component_weight
        for sym, w in self._last_b.items():
            combined[sym] = combined.get(sym, 0.0) + w * self.component_weight
        for sym, w in self._last_c.items():
            combined[sym] = combined.get(sym, 0.0) + w * self.component_weight

        if not combined:
            return []

        # Normalize to exposure cap
        total_w = sum(combined.values())
        if total_w > self.exposure_cap:
            scale = self.exposure_cap / total_w
            combined = {sym: w * scale for sym, w in combined.items()}

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in combined and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in combined.items():
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


NAME = "ensemble_quality_termstruct_seasonal"
HYPOTHESIS = (
    "Ensemble of 3 orthogonal strategies: nearhi_momentum_quality (quality equity momentum, 1/3) "
    "+ bond_termstruct_curve_rotation (yield-curve bond duration, 1/3) + halloween_sell_in_may "
    "(seasonal SPY/TLT, 1/3); combines orthogonal signals (stock selection, rates, seasonality); "
    "overlapping positions netted; exposure capped 0.97"
)

STRATEGY = EnsembleQualityTermstructSeasonal()
