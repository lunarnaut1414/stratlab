"""Ensemble of Rate-Trend Gate + Breadth Rotation + Always-Invested RP — gen_8 opus-3

Equal-weight ensemble of 3 STRUCTURALLY ORTHOGONAL components:

  A. TNX 200d Yield Trend Gate
     - SPY in bear (SPY < SPY 200d MA): TLT 100%
     - TNX above its 200d MA (rising/elevated rates regime): SPY 62% + TLT 38%
     - TNX below its 200d MA (accommodative rate regime) AND SPY bull: top-15
       SP500 stocks by 63d momentum.
     - Signal family: MACRO RATE LEVEL (TNX vs its own trend).
     - Source strategy: gen8_tnx_yield_trend_equity_gate (IS Calmar 1.01, #1)

  B. RSP-SPY 42d Breadth Rotation
     - SPY bear (SPY < SPY 200d MA): TLT 100%
     - RSP 42d return > SPY 42d return (broad participation): QQQ 100%
     - SPY leads RSP (narrow leadership): SPY 62% + IEF 38%
     - Signal family: INTRA-EQUITY BREADTH STRUCTURE (cap-weight vs equal-weight).
     - Source strategy: gen8_rsp_spy_breadth_qqq_rotation (IS Calmar 0.79, h1/h2
       0.88/0.92 — most stable on leaderboard).

  C. Inverse-Volatility Risk Parity QQQ/GLD (no bonds)
     - Always invested. No regime gate. Weights proportional to 1 / 20d-stddev
       of daily returns of QQQ and GLD, normalized to sum to 1.
     - Signal family: NONE (pure portfolio construction; no timing).
     - DELIBERATELY excludes any bond ETF to avoid correlation with the
       bond-termstruct cluster (an earlier v1 with SPY/IEF/GLD hit 0.85+ corr
       to gen6_bond_termstruct_curve_rotation due to inv-vol overweighting
       IEF). QQQ + GLD as the two assets gives equity-growth + inflation-hedge
       diversification with zero bond loading.

Why this ensemble is meaningfully DIFFERENT from gen8_ensemble_credit_volcarry_seasonal
(the existing #2 ensemble at IS Calmar 0.97):

  Existing ensemble uses:    Mine uses:
    A: JNK 30d MA (credit)     A: TNX 200d MA (rate level)
    B: SPY 21d RV pctile       B: RSP/SPY 42d breadth ratio
    C: Calendar (Halloween)    C: Always-invested 20d inv-vol RP

  No shared inputs between the two ensembles. Mine has an always-invested
  component (C) which the existing ensemble lacks — this should reduce loss
  during regime-gate whipsaws.

Pairwise structural-orthogonality argument (corr_dump not available):
  - A reads ^TNX rolling level; B reads RSP & SPY returns; C reads SPY/IEF/GLD
    realized vol. None of the three reads any series the other two read.
  - A and B share an SPY bear gate (SPY 200d MA), creating a partial co-move
    in deep bear: both go to TLT. This is INTENTIONAL — tail-risk overlap is
    desirable. In benign regimes (SPY bull) their internal allocations
    diverge: A picks SP500 stocks or SPY+TLT by rate level; B picks QQQ or
    SPY+IEF by breadth — these are different equity-tilt choices.
  - C is regime-agnostic and provides drawdown ballast in regimes where both
    A and B are stuck in their less-favorable branches.

Composition rule: each component sized at COMPONENT_WEIGHT = 1/3, target
weights summed per symbol, total exposure capped at EXPOSURE_CAP = 0.97.
Overlapping ETF positions net naturally (e.g. SPY held by both A and B in
some regimes; TLT by both A and B in bear).

IS window: 2010-01-01 to 2018-12-31. Tested via standard submit harness.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — TNX 200d yield trend gate
# ---------------------------------------------------------------------------
A_MOMENTUM_WINDOW = 63       # for SP500 momentum selection in low-rate regime
A_YIELD_TREND_WINDOW = 200   # TNX 200d MA
A_SPY_TREND_WINDOW = 200     # SPY bear gate
A_TOP_K = 15                 # top-15 SP500 stocks in risk-on branch
A_SPY_BLEND = 0.618          # rising-rate branch SPY share
A_TLT_BLEND = 0.382          # rising-rate branch TLT share

# ---------------------------------------------------------------------------
# Component B — RSP/SPY breadth rotation
# ---------------------------------------------------------------------------
B_BREADTH_WINDOW = 42        # RSP vs SPY 42d return delta
B_SPY_TREND_WINDOW = 200     # SPY bear gate (shared structurally with A)
B_SPY_BLEND = 0.618          # narrow-leadership SPY share
B_IEF_BLEND = 0.382          # narrow-leadership IEF share

# ---------------------------------------------------------------------------
# Component C — Inverse-vol risk parity QQQ/GLD (always invested, NO bonds)
# ---------------------------------------------------------------------------
C_VOL_WINDOW = 20            # 20d realized vol for inv-vol weighting
C_BASE_ASSETS = ("QQQ", "GLD")

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0   # each component sized to 1/3 of capital
EXPOSURE_CAP = 0.97             # total invested cap
REBALANCE_EVERY = 5             # weekly ensemble rebalance
WARMUP_BARS = 215               # ~200d trend windows + margin
MIN_TRADE_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """TNX yield trend gate (top-15 SP500 mom / SPY+TLT blend / TLT)."""
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

    # TNX yield trend (default low-rate if unavailable)
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

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live_syms = set(closes_now.index)

    weights: dict[str, float] = {}

    if not spy_bull:
        # Bear: fully TLT
        if "TLT" in live_syms:
            weights["TLT"] = 1.0
        return weights if weights else None

    if not tnx_low_rate:
        # Rising/elevated rates: SPY+TLT blend
        if "SPY" in live_syms:
            weights["SPY"] = A_SPY_BLEND
        if "TLT" in live_syms:
            weights["TLT"] = A_TLT_BLEND
        return weights if weights else None

    # Low-rate + SPY bull: top-K SP500 by 63d momentum
    prices = ctx.closes_window(A_MOMENTUM_WINDOW + 5)
    if len(prices) < A_MOMENTUM_WINDOW:
        if "SPY" in live_syms:
            return {"SPY": 1.0}
        return None

    scores: dict[str, float] = {}
    excluded = {"SPY", "TLT", "IEF", "GLD", "QQQ", "RSP", "JNK", "LQD"}
    for sym in prices.columns:
        if sym in excluded:
            continue
        col = prices[sym].dropna()
        if len(col) < A_MOMENTUM_WINDOW:
            continue
        past = float(col.iloc[-A_MOMENTUM_WINDOW])
        if past <= 0:
            continue
        ret = float(col.iloc[-1] / past - 1.0)
        if np.isfinite(ret):
            scores[sym] = ret

    if len(scores) < 5:
        if "SPY" in live_syms:
            return {"SPY": 1.0}
        return None

    k = min(A_TOP_K, len(scores))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    per = 1.0 / len(ranked)
    for sym in ranked:
        if sym in live_syms:
            weights[sym] = per
    return weights if weights else None


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """RSP/SPY 42d breadth rotation (QQQ / SPY+IEF / TLT)."""
    if ctx.idx < B_SPY_TREND_WINDOW + 10:
        return None
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < B_SPY_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-B_SPY_TREND_WINDOW:].mean())
    spy_now = float(spy_close.iloc[-1])
    spy_bull = spy_now > spy_sma

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live_syms = set(closes_now.index)

    weights: dict[str, float] = {}

    if not spy_bull:
        if "TLT" in live_syms:
            weights["TLT"] = 1.0
        return weights if weights else None

    # RSP vs SPY 42d return
    broad_participation = True
    try:
        rsp_hist = ctx.history("RSP")
        if (rsp_hist is not None and
                len(rsp_hist) >= B_BREADTH_WINDOW + 2 and
                len(spy_close) >= B_BREADTH_WINDOW + 2):
            rsp_close = rsp_hist["close"].dropna()
            if (len(rsp_close) >= B_BREADTH_WINDOW + 1 and
                    len(spy_close) >= B_BREADTH_WINDOW + 1):
                rsp_past = float(rsp_close.iloc[-B_BREADTH_WINDOW - 1])
                spy_past = float(spy_close.iloc[-B_BREADTH_WINDOW - 1])
                if rsp_past > 0 and spy_past > 0:
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_past - 1.0)
                    spy_ret = float(spy_close.iloc[-1] / spy_past - 1.0)
                    if np.isfinite(rsp_ret) and np.isfinite(spy_ret):
                        broad_participation = rsp_ret > spy_ret
    except Exception:
        pass

    if broad_participation:
        if "QQQ" in live_syms:
            weights["QQQ"] = 1.0
        elif "SPY" in live_syms:
            weights["SPY"] = 1.0
    else:
        if "SPY" in live_syms:
            weights["SPY"] = B_SPY_BLEND
        if "IEF" in live_syms:
            weights["IEF"] = B_IEF_BLEND

    return weights if weights else None


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Always-invested inverse-vol risk parity SPY/IEF/GLD."""
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


class EnsembleRateBreadthRP(Strategy):
    """Equal-weight ensemble of TNX trend gate + RSP/SPY breadth + always-invested RP."""

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

        # Aggregate: each component contributes its (already-normalized within-
        # component) weights * COMPONENT_WEIGHT. Symbols held by multiple
        # components net (e.g. SPY in both A's blend and B's blend).
        target: dict[str, float] = {}
        for comp in (a_w, b_w, c_w):
            if comp is None:
                continue
            for sym, w in comp.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

        if not target:
            return []

        # Apply exposure cap by scaling down if total > cap
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

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Reduce overweight positions
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        # Increase underweight positions
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


def _universe() -> list[str]:
    """SP500 (for component A momentum) plus the ETFs and signal series."""
    from stratlab.data.universe import sp500_tickers
    extras = ["SPY", "TLT", "IEF", "GLD", "QQQ", "RSP", "^TNX"]
    return sp500_tickers() + extras


NAME = "opus3_ensemble_rate_breadth_rp"
HYPOTHESIS = (
    "Ensemble of 3 orthogonal-family components: "
    "A=TNX 200d yield trend gate (top-15 SP500 mom / SPY+TLT / TLT bear), "
    "B=RSP-SPY 42d breadth (QQQ / SPY+IEF / TLT bear), "
    "C=always-invested inverse-vol RP on SPY/IEF/GLD; "
    "each 1/3 capital, overlapping ETFs netted, exposure capped 0.97, weekly rebalance; "
    "structurally distinct from existing ensemble_credit_volcarry_seasonal "
    "(no JNK, no vol-carry, no calendar; includes always-invested RP component)"
)

UNIVERSE = _universe

STRATEGY = EnsembleRateBreadthRP()
