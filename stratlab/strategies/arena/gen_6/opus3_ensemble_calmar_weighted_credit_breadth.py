"""Calmar-weighted ensemble of 3 low-LMC stable-h2 gen_6 diversifiers.

Unlike opus3_ensemble_credit_sector_breadth (equal-weight 1/3), this
ensemble tilts capital toward components with higher and more stable
IS Calmar.

Components and weights (allocated fractions sum to 1.0):

  W = 0.40 (largest) — jnk_vix_dual_gate_qqq (sonnet-6, gen_6)
      Signals: JNK 20d MA trend + VIX <20/<28 thresholds.
      Tier 1 risk-on (JNK bull AND VIX<20)  -> QQQ 100%
      Tier 2 (JNK bull AND 20<=VIX<28)      -> SPY 100%
      Tier 3 (JNK bull AND VIX>=28)         -> SPY 60% + TLT 40%
      Risk-off (JNK bear)                    -> SHY 50% + TLT 50%
      Leaderboard: IS Calmar 0.86, h1=0.83, h2=1.14 (improving),
                   LMC=0.66, corr_to_top5=0.71.

  W = 0.35 — hy_credit_qqq_rotation (sonnet-10, gen_6)
      Signal: JNK > 30d MA AND SPY > 100d MA.
      Risk-on  -> QQQ 100%
      Risk-off -> TLT 100%
      Leaderboard: IS Calmar 0.78, h1=0.63, h2=1.30 (improving),
                   LMC=0.50, corr_to_top5=0.41 (lowest in set).

  W = 0.25 (smallest) — smallcap_leadership_rotation (sonnet-9, gen_6)
      Signal: IWM 20d return > SPY 20d return.
      Risk-on  -> QQQ 60% + IWM 40%
      Risk-off -> SPY 60% + TLT 40%
      Leaderboard: IS Calmar 0.61, h1=0.76, h2=0.58 (stable),
                   LMC=0.28 (lowest in set), corr_to_top5=0.60.

Why these weights:
  - First two are credit-trend-based (orthogonal secondaries: VIX vs SPY
    trend) and have higher Calmar — they get more capital.
  - Third is purely an equity-size signal (no credit, no vol) with the
    lowest loss_mode_corr_to_top5; it acts as a diversification ballast
    even at smaller weight.
  - Sum of weights = 1.0; total exposure capped at 0.97 after netting.

Why this triplet:
  - All three components have h2 >= 0.58 (avoid brief's flagged fragile
    components: risk_parity_4asset_vnq h2=0.33, qqq_vs_xlu_rotation
    h2=0.29, factor_etf_rotation h2=0.38).
  - Two of three improve h2 vs h1 — sturdy second-half performance.
  - Average corr_to_top5 weighted by W = 0.40*0.71 + 0.35*0.41 +
    0.25*0.60 = 0.578.
  - Average loss_mode_corr_to_top5 weighted = 0.40*0.66 + 0.35*0.50 +
    0.25*0.28 = 0.508 (well below leader cluster's ~0.93).

Hypothesis: weighting components by IS Calmar (informally, by their
edge magnitude) should beat equal-weight when the highest-Calmar
component is also h2-stable. Because both top components are
credit-allocators with improving h2, this favors the strongest signal
in the IS regime tail.

Pattern follows gen5_ensemble_bond_credit_seasonal.py.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — jnk_vix_dual_gate_qqq parameters
# ---------------------------------------------------------------------------
A_JNK_MA = 20
A_VIX_CALM = 20.0
A_VIX_CAUTION = 28.0
A_WEIGHT = 0.40

# ---------------------------------------------------------------------------
# Component B — hy_credit_qqq_rotation parameters
# ---------------------------------------------------------------------------
B_JNK_MA = 30
B_SPY_MA = 100
B_WEIGHT = 0.35

# ---------------------------------------------------------------------------
# Component C — smallcap_leadership_rotation parameters
# ---------------------------------------------------------------------------
C_RS_WINDOW = 20
C_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
EXPOSURE_CAP = 0.97
ENSEMBLE_REBALANCE_EVERY = 5
WARMUP_BARS = 110
MIN_TRADE_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component A: jnk_vix_dual_gate_qqq raw target weights (sum 1.0)."""
    warmup = A_JNK_MA + 10
    if ctx.idx < warmup:
        return None

    try:
        jnk_hist = ctx.history("JNK")
    except KeyError:
        return None
    jnk_close = jnk_hist["close"].dropna() if jnk_hist is not None else None
    if jnk_close is None or len(jnk_close) < A_JNK_MA + 1:
        return None
    jnk_sma = float(jnk_close.iloc[-A_JNK_MA:].mean())
    credit_bullish = float(jnk_close.iloc[-1]) > jnk_sma

    try:
        vix_hist = ctx.history("^VIX")
    except KeyError:
        return None
    if vix_hist is None or len(vix_hist) < 1:
        return None
    vix_close = vix_hist["close"].dropna()
    if len(vix_close) < 1:
        return None
    vix_level = float(vix_close.iloc[-1])

    closes_now = ctx.closes()

    if not credit_bullish:
        weights: dict[str, float] = {}
        if "SHY" in closes_now.index:
            weights["SHY"] = 0.50
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.50
        return weights if weights else None

    if vix_level < A_VIX_CALM:
        return {"QQQ": 1.0} if "QQQ" in closes_now.index else (
            {"SPY": 1.0} if "SPY" in closes_now.index else None)
    if vix_level < A_VIX_CAUTION:
        return {"SPY": 1.0} if "SPY" in closes_now.index else (
            {"QQQ": 1.0} if "QQQ" in closes_now.index else None)
    weights = {}
    if "SPY" in closes_now.index:
        weights["SPY"] = 0.60
    if "TLT" in closes_now.index:
        weights["TLT"] = 0.40
    return weights if weights else None


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component B: hy_credit_qqq_rotation raw target weights (full 1.0)."""
    warmup = max(B_JNK_MA, B_SPY_MA) + 10
    if ctx.idx < warmup:
        return None

    try:
        jnk_hist = ctx.history("JNK")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    jnk_close = jnk_hist["close"].dropna() if jnk_hist is not None else None
    spy_close = spy_hist["close"].dropna() if spy_hist is not None else None
    if jnk_close is None or len(jnk_close) < B_JNK_MA + 1:
        return None
    if spy_close is None or len(spy_close) < B_SPY_MA + 1:
        return None

    jnk_sma = float(jnk_close.iloc[-B_JNK_MA:].mean())
    jnk_bull = float(jnk_close.iloc[-1]) > jnk_sma

    spy_sma = float(spy_close.iloc[-B_SPY_MA:].mean())
    spy_bull = float(spy_close.iloc[-1]) > spy_sma

    closes_now = ctx.closes()
    if jnk_bull and spy_bull:
        return {"QQQ": 1.0} if "QQQ" in closes_now.index else None
    return {"TLT": 1.0} if "TLT" in closes_now.index else None


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component C: smallcap_leadership_rotation raw target weights."""
    if ctx.idx < C_RS_WINDOW + 5:
        return None

    try:
        iwm_hist = ctx.history("IWM")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    iwm_close = iwm_hist["close"].dropna() if iwm_hist is not None else None
    spy_close = spy_hist["close"].dropna() if spy_hist is not None else None
    if iwm_close is None or len(iwm_close) < C_RS_WINDOW + 1:
        return None
    if spy_close is None or len(spy_close) < C_RS_WINDOW + 1:
        return None
    iwm_ret = float(iwm_close.iloc[-1] / iwm_close.iloc[-C_RS_WINDOW - 1] - 1.0)
    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-C_RS_WINDOW - 1] - 1.0)
    if not (np.isfinite(iwm_ret) and np.isfinite(spy_ret)):
        return None

    closes_now = ctx.closes()
    weights: dict[str, float] = {}
    if iwm_ret > spy_ret:
        if "QQQ" in closes_now.index:
            weights["QQQ"] = 0.60
        if "IWM" in closes_now.index:
            weights["IWM"] = 0.40
    else:
        if "SPY" in closes_now.index:
            weights["SPY"] = 0.60
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.40
    return weights if weights else None


class EnsembleCalmarWeightedCreditBreadth(Strategy):
    """Calmar-weighted ensemble (W=0.40/0.35/0.25)."""

    def __init__(
        self,
        rebalance_every: int = ENSEMBLE_REBALANCE_EVERY,
        exposure_cap: float = EXPOSURE_CAP,
        a_weight: float = A_WEIGHT,
        b_weight: float = B_WEIGHT,
        c_weight: float = C_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            exposure_cap=exposure_cap,
            a_weight=a_weight,
            b_weight=b_weight,
            c_weight=c_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.exposure_cap = float(exposure_cap)
        self.a_weight = float(a_weight)
        self.b_weight = float(b_weight)
        self.c_weight = float(c_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        a_w = _component_a_weights(ctx)
        b_w = _component_b_weights(ctx)
        c_w = _component_c_weights(ctx)

        target: dict[str, float] = {}
        for component_targets, w in (
            (a_w, self.a_weight),
            (b_w, self.b_weight),
            (c_w, self.c_weight),
        ):
            if component_targets is None:
                continue
            for sym, raw in component_targets.items():
                target[sym] = target.get(sym, 0.0) + raw * w

        total_w = sum(target.values())
        if total_w > self.exposure_cap and total_w > 0.0:
            scale = self.exposure_cap / total_w
            target = {sym: w * scale for sym, w in target.items()}

        if not target:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
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
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta < -MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        for sym, tgt in target_shares.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta > MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


NAME = "opus3_ensemble_calmar_weighted_credit_breadth"
HYPOTHESIS = (
    "Calmar-weighted ensemble of 3 low-LMC gen6 diversifiers: "
    "jnk_vix_dual_gate_qqq (h1=0.83 h2=1.14 stable, weight 0.40), "
    "hy_credit_qqq_rotation (h1=0.63 h2=1.30 improving, weight 0.35), "
    "smallcap_leadership_rotation (h1=0.76 h2=0.58 stable, weight 0.25). "
    "Each component contributes target weights * its weight; overlapping "
    "positions netted; total exposure capped 0.97. Tilts toward "
    "higher-Calmar credit allocators while breadth component diversifies "
    "against credit-cycle-shared loss modes."
)
UNIVERSE = ["JNK", "SPY", "QQQ", "TLT", "SHY", "IWM", "^VIX"]

STRATEGY = EnsembleCalmarWeightedCreditBreadth()
