"""Ensemble of Rate-Gated Factor + SPY/GLD Seasonal + IWM/QQQ/GLD RP — gen_8 opus-3 #2

DESIGN GOAL: Build an ensemble that is structurally orthogonal to the SPY-heavy
top-5 cluster by deliberately MINIMIZING SPY exposure across all components.
This is a counter-positioning ensemble vs the leaderboard, intended to fill the
diversification niche that pure-SPY-rotation strategies leave open.

Components (each equal-weight at 1/3 of capital):

  A. TNX 200d Yield Trend Factor Gate (factor ETF sleeve, NO SP500 stock picking)
     - SPY < SPY 200d MA (deep bear): TLT 100% (only universal fallback)
     - TNX > TNX 200d MA (rising rates regime): MTUM 100%
       (momentum factor outperforms in rising-rate equity regimes 2013-2015 style)
     - TNX < TNX 200d MA AND SPY bull (accommodative): IWM 100%
       (small-caps lead in accommodative regimes; broadens to non-SPY equity)
     - Signal family: MACRO RATE LEVEL. Inspired by gen8_tnx_yield_trend_equity_gate
       but routed to factor ETFs (MTUM/IWM) instead of top-K SP500 momentum.
     - Routing change deliberate: removes SP500-stock-picking overlap with most
       top-5 strategies.

  B. Halloween Seasonal SPY/GLD (NOT SPY/TLT — bond-free)
     - Nov-Apr (winter risk-on): SPY 100%
     - May-Oct (summer): GLD 100%
     - Signal family: CALENDAR ONLY (zero market data).
     - Departure from gen5 ensemble: summer side is GLD not TLT, which removes
       bond exposure from the seasonal sleeve entirely.

  C. Inverse-Vol RP on IWM / QQQ / GLD (no SPY, no bonds, always invested)
     - 20d inverse-realized-vol weights, normalized to sum to 1.
     - Signal family: NONE (pure portfolio construction).
     - IWM and QQQ have lower daily-corr to SPY than SPY-itself, and GLD is
       cross-asset. This component intentionally avoids the SPY+bond mix that
       dominates the existing ensemble (and that caused v1 of this ensemble
       to corr 0.85 with the bond-termstruct cluster).

Why this is structurally distinct from gen8_ensemble_credit_volcarry_seasonal (0.97):
  - That ensemble uses JNK credit + SPY 21d RV + Halloween-SPY/TLT.
  - Mine uses TNX rate level + Halloween-SPY/GLD + IWM/QQQ/GLD RP.
  - No shared input signal. Mine specifically removes SPY-heavy bond pairs.

Pairwise structural-orthogonality argument (corr_dump not available):
  - A reads ^TNX rolling level + SPY 200d gate; rotates between MTUM/IWM/TLT.
  - B reads calendar month only.
  - C reads IWM/QQQ/GLD realized vol; always invested.
  - The only overlap is the SPY 200d gate inside A (used as a deep-bear shutoff
    to TLT) — this is a tail-risk safety, intentional, and shared with B/C only
    in the sense that during a deep bear all components struggle.

IS window: 2010-01-01 to 2018-12-31.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — TNX 200d gate routing to factor ETFs (MTUM/IWM/TLT)
# ---------------------------------------------------------------------------
A_YIELD_TREND_WINDOW = 200
A_SPY_TREND_WINDOW = 200

# ---------------------------------------------------------------------------
# Component B — Halloween seasonal SPY (winter) / GLD (summer)
# ---------------------------------------------------------------------------
B_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

# ---------------------------------------------------------------------------
# Component C — Inverse-vol RP IWM/QQQ/GLD (always invested)
# ---------------------------------------------------------------------------
C_VOL_WINDOW = 20
C_BASE_ASSETS = ("IWM", "QQQ", "GLD")

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97
REBALANCE_EVERY = 5
WARMUP_BARS = 215
MIN_TRADE_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """TNX 200d gate routing to MTUM (rising rates) / IWM (low rates) / TLT (bear)."""
    if ctx.idx < A_SPY_TREND_WINDOW + 10:
        return None
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < A_SPY_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-A_SPY_TREND_WINDOW:].mean())
    spy_now = float(spy_close.iloc[-1])
    spy_bull = spy_now > spy_sma

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live_syms = set(closes_now.index)

    if not spy_bull:
        if "TLT" in live_syms:
            return {"TLT": 1.0}
        return None

    # TNX vs 200d MA
    tnx_low_rate = True
    try:
        tnx_hist = ctx.history("^TNX")
        if tnx_hist is not None and len(tnx_hist) >= A_YIELD_TREND_WINDOW + 2:
            tnx_close = tnx_hist["close"].dropna()
            if len(tnx_close) >= A_YIELD_TREND_WINDOW:
                tnx_ma = float(tnx_close.iloc[-A_YIELD_TREND_WINDOW:].mean())
                tnx_now_val = float(tnx_close.iloc[-1])
                tnx_low_rate = tnx_now_val < tnx_ma
    except Exception:
        pass

    if tnx_low_rate:
        # Accommodative rates: small-cap leadership
        if "IWM" in live_syms:
            return {"IWM": 1.0}
        if "SPY" in live_syms:
            return {"SPY": 1.0}
    else:
        # Rising rates: momentum factor
        if "MTUM" in live_syms:
            return {"MTUM": 1.0}
        if "SPY" in live_syms:
            return {"SPY": 1.0}
    return None


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """Halloween seasonal: SPY in Nov-Apr, GLD in May-Oct."""
    if ctx.idx < 5:
        return None
    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    month = ctx.timestamp.month
    if month in B_WINTER_MONTHS:
        if "SPY" in closes_now.index:
            return {"SPY": 1.0}
    else:
        if "GLD" in closes_now.index:
            return {"GLD": 1.0}
    return None


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Always-invested inverse-vol RP on IWM/QQQ/GLD."""
    if ctx.idx < C_VOL_WINDOW + 10:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live_syms = set(closes_now.index)

    inv_vols: dict[str, float] = {}
    for sym in C_BASE_ASSETS:
        if sym not in live_syms:
            continue
        try:
            hist = ctx.history(sym)
        except KeyError:
            continue
        close = hist["close"].dropna()
        if len(close) < C_VOL_WINDOW + 2:
            continue
        tail = close.iloc[-(C_VOL_WINDOW + 1):]
        if (tail <= 0).any():
            continue
        log_rets = np.log(tail.values[1:] / tail.values[:-1])
        vol = float(np.std(log_rets))
        if vol > 1e-8 and np.isfinite(vol):
            inv_vols[sym] = 1.0 / vol

    if not inv_vols:
        return None

    total = sum(inv_vols.values())
    if total <= 0:
        return None

    return {sym: iv / total for sym, iv in inv_vols.items()}


class EnsembleFactorSeasonalAlt(Strategy):
    """Anti-SPY-correlation ensemble: rate-gated factor + Halloween SPY/GLD + IWM/QQQ/GLD RP."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
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

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        a_w = _component_a_weights(ctx)
        b_w = _component_b_weights(ctx)
        c_w = _component_c_weights(ctx)

        target: dict[str, float] = {}
        for comp in (a_w, b_w, c_w):
            if comp is None:
                continue
            for sym, w in comp.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

        if not target:
            return []

        total_w = sum(target.values())
        if total_w > self.exposure_cap and total_w > 0.0:
            scale = self.exposure_cap / total_w
            target = {sym: w * scale for sym, w in target.items()}

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target_shares: dict[str, int] = {}
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            shares = int(equity * weight / price)
            if shares > 0:
                target_shares[sym] = shares

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


UNIVERSE = ["SPY", "QQQ", "IWM", "MTUM", "TLT", "GLD", "^TNX"]

NAME = "opus3_ensemble_factor_seasonal_alt"
HYPOTHESIS = (
    "Anti-SPY-correlation ensemble: "
    "A=TNX 200d yield trend gate routing to MTUM (rising rates) / IWM (accommodative) / TLT (SPY bear); "
    "B=Halloween seasonal SPY (Nov-Apr) / GLD (May-Oct, NOT TLT — drops bond exposure); "
    "C=always-invested inv-vol RP on IWM/QQQ/GLD (no SPY, no bonds); "
    "each 1/3 capital, exposure cap 0.97, weekly rebalance; "
    "deliberate counter-positioning to SPY-heavy top-5 cluster"
)

STRATEGY = EnsembleFactorSeasonalAlt()
