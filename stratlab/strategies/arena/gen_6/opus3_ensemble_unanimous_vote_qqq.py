"""Unanimous-vote tiered ensemble of 3 risk-on gates from gen_6.

Each component independently emits a binary risk-on / risk-off vote based
on a structurally orthogonal signal. Allocation tier scales with the count
of risk-on votes.

Components and signals:

  V1. jnk_vix_dual_gate_qqq (sonnet-6)
      Risk-on iff JNK > 20d MA AND VIX < 20.
      Leaderboard: IS Calmar 0.86, h1=0.83, h2=1.14, LMC=0.66.

  V2. hy_credit_qqq_rotation (sonnet-10)
      Risk-on iff JNK > 30d MA AND SPY > 100d MA.
      Leaderboard: IS Calmar 0.78, h1=0.63, h2=1.30, LMC=0.50.

  V3. smallcap_leadership_rotation (sonnet-9)
      Risk-on iff IWM 20d return > SPY 20d return.
      Leaderboard: IS Calmar 0.61, h1=0.76, h2=0.58, LMC=0.28.

Signal orthogonality:
  V1 = JNK trend AND volatility level (credit + vol)
  V2 = JNK trend AND SPY trend (credit + equity trend)
  V3 = small vs large equity relative momentum (size factor)

  V1 and V2 share the JNK trend signal but combine it with very different
  secondaries (VIX level vs SPY trend). V3 is purely an equity size signal,
  independent of credit and vol.

Allocation rule (count of risk-on votes from {V1, V2, V3}):
  3 / 3 risk-on -> QQQ 97% (unanimous bull)
  2 / 3 risk-on -> QQQ 60% + SPY 37% (majority bull, hedged)
  1 / 3 risk-on -> SPY 60% + TLT 37% (caution)
  0 / 3 risk-on -> SHY 50% + TLT 47% (full risk-off)

Why this ensemble structure (vs equal-weight):
  - Equal-weight averages weights bar-by-bar. Voting averages SIGNALS — a
    single component's spurious risk-on doesn't drive 1/3 of capital into
    QQQ. Both modes have merit; this fills the voting niche.
  - Unanimity is a strict quality bar. Combined with graceful degradation
    tiers, it produces a different allocation pattern than any constituent.
  - All three components have h2 >= 0.58 — ensemble doesn't carry fragile
    components flagged in the brief (risk_parity_4asset_vnq h2=0.33,
    qqq_vs_xlu_rotation h2=0.29).

Hypothesis: structurally orthogonal credit/vol/breadth confirmation
should reduce false-positive risk-on signals, lowering drawdowns at the
cost of some upside capture. Net Calmar should be competitive with the
best single constituent while loss_mode_corr_to_top5 stays well below
the leaderboard cluster.

Pattern follows gen5_ensemble_bond_credit_seasonal.py.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component V1 — jnk_vix_dual_gate parameters
# ---------------------------------------------------------------------------
V1_JNK_MA = 20
V1_VIX_CALM = 20.0

# ---------------------------------------------------------------------------
# Component V2 — hy_credit_qqq_rotation parameters
# ---------------------------------------------------------------------------
V2_JNK_MA = 30
V2_SPY_MA = 100

# ---------------------------------------------------------------------------
# Component V3 — smallcap_leadership_rotation parameters
# ---------------------------------------------------------------------------
V3_RS_WINDOW = 20

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
EXPOSURE_CAP = 0.97
ENSEMBLE_REBALANCE_EVERY = 5  # weekly
WARMUP_BARS = 110
MIN_TRADE_DELTA = 1


def _v1_risk_on(ctx: BarContext) -> bool | None:
    """V1: JNK > 20d MA AND VIX < 20."""
    if ctx.idx < V1_JNK_MA + 10:
        return None
    try:
        jnk_hist = ctx.history("JNK")
    except KeyError:
        return None
    jnk_close = jnk_hist["close"].dropna() if jnk_hist is not None else None
    if jnk_close is None or len(jnk_close) < V1_JNK_MA + 1:
        return None
    jnk_sma = float(jnk_close.iloc[-V1_JNK_MA:].mean())
    jnk_bull = float(jnk_close.iloc[-1]) > jnk_sma

    try:
        vix_hist = ctx.history("^VIX")
    except KeyError:
        return None
    if vix_hist is None or len(vix_hist) < 1:
        return None
    vix_close = vix_hist["close"].dropna()
    if len(vix_close) < 1:
        return None
    vix_now = float(vix_close.iloc[-1])
    vix_calm = vix_now < V1_VIX_CALM

    return jnk_bull and vix_calm


def _v2_risk_on(ctx: BarContext) -> bool | None:
    """V2: JNK > 30d MA AND SPY > 100d MA."""
    if ctx.idx < max(V2_JNK_MA, V2_SPY_MA) + 10:
        return None

    try:
        jnk_hist = ctx.history("JNK")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    jnk_close = jnk_hist["close"].dropna() if jnk_hist is not None else None
    spy_close = spy_hist["close"].dropna() if spy_hist is not None else None
    if jnk_close is None or len(jnk_close) < V2_JNK_MA + 1:
        return None
    if spy_close is None or len(spy_close) < V2_SPY_MA + 1:
        return None

    jnk_sma = float(jnk_close.iloc[-V2_JNK_MA:].mean())
    jnk_bull = float(jnk_close.iloc[-1]) > jnk_sma

    spy_sma = float(spy_close.iloc[-V2_SPY_MA:].mean())
    spy_bull = float(spy_close.iloc[-1]) > spy_sma

    return jnk_bull and spy_bull


def _v3_risk_on(ctx: BarContext) -> bool | None:
    """V3: IWM 20d return > SPY 20d return."""
    if ctx.idx < V3_RS_WINDOW + 5:
        return None
    try:
        iwm_hist = ctx.history("IWM")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    iwm_close = iwm_hist["close"].dropna() if iwm_hist is not None else None
    spy_close = spy_hist["close"].dropna() if spy_hist is not None else None
    if iwm_close is None or len(iwm_close) < V3_RS_WINDOW + 1:
        return None
    if spy_close is None or len(spy_close) < V3_RS_WINDOW + 1:
        return None
    iwm_ret = float(iwm_close.iloc[-1] / iwm_close.iloc[-V3_RS_WINDOW - 1] - 1.0)
    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-V3_RS_WINDOW - 1] - 1.0)
    if not (np.isfinite(iwm_ret) and np.isfinite(spy_ret)):
        return None
    return iwm_ret > spy_ret


def _allocation_for_votes(votes: int, available: set[str]) -> dict[str, float]:
    """Map vote count to target weights (raw, before exposure cap)."""
    if votes == 3 and "QQQ" in available:
        return {"QQQ": 1.00}
    if votes == 2:
        if "QQQ" in available and "SPY" in available:
            return {"QQQ": 0.60, "SPY": 0.40}
        if "SPY" in available:
            return {"SPY": 1.00}
        if "QQQ" in available:
            return {"QQQ": 1.00}
    if votes == 1:
        if "SPY" in available and "TLT" in available:
            return {"SPY": 0.60, "TLT": 0.40}
        if "SPY" in available:
            return {"SPY": 1.00}
    if votes == 0:
        if "SHY" in available and "TLT" in available:
            return {"SHY": 0.50, "TLT": 0.50}
        if "TLT" in available:
            return {"TLT": 1.00}
        if "SHY" in available:
            return {"SHY": 1.00}
    return {}


class EnsembleUnanimousVoteQqq(Strategy):
    """Voting ensemble of V1+V2+V3 risk-on gates with tiered allocation."""

    def __init__(
        self,
        rebalance_every: int = ENSEMBLE_REBALANCE_EVERY,
        exposure_cap: float = EXPOSURE_CAP,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            exposure_cap=exposure_cap,
        )
        self.rebalance_every = int(rebalance_every)
        self.exposure_cap = float(exposure_cap)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        v1 = _v1_risk_on(ctx)
        v2 = _v2_risk_on(ctx)
        v3 = _v3_risk_on(ctx)
        # If any vote is unavailable (warmup not done) treat as risk-off
        v1_b = bool(v1) if v1 is not None else False
        v2_b = bool(v2) if v2 is not None else False
        v3_b = bool(v3) if v3 is not None else False

        votes = int(v1_b) + int(v2_b) + int(v3_b)

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        available = set(closes_now.index)

        raw_weights = _allocation_for_votes(votes, available)
        if not raw_weights:
            return []

        # Apply exposure cap (raw weights sum to 1.0; scale to cap)
        target = {sym: w * self.exposure_cap for sym, w in raw_weights.items()}

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

        # 1. Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # 2. Reduce overweights
        for sym, tgt in target_shares.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta < -MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        # 3. Buys
        for sym, tgt in target_shares.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta > MIN_TRADE_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


NAME = "opus3_ensemble_unanimous_vote_qqq"
HYPOTHESIS = (
    "Unanimous-vote QQQ ensemble of 3 risk-on gates: jnk_vix_dual_gate_qqq "
    "(JNK trend AND VIX<20), hy_credit_qqq_rotation (JNK trend AND SPY>100d), "
    "smallcap_leadership_rotation (IWM>SPY 20d). All 3 risk-on -> QQQ 97; "
    "exactly 2 risk-on -> QQQ 60% + SPY 37%; exactly 1 -> SPY 60% + TLT 37%; "
    "0 risk-on -> SHY 50% + TLT 47%; weekly rebalance. Conjunction of "
    "orthogonal credit/vol/breadth signals raises bar for full risk-on but "
    "adds graceful degradation tiers absent in any single component."
)
UNIVERSE = ["JNK", "SPY", "QQQ", "TLT", "SHY", "IWM", "^VIX"]

STRATEGY = EnsembleUnanimousVoteQqq()
