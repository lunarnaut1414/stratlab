"""Equal-weight ensemble of 3 stable-h2, low-LMC gen_6 survivors.

Components (each sized 1/3 of capital, target weights summed and netted):

  A. hy_credit_qqq_rotation (sonnet-10, gen_6)
     - QQQ when JNK > 30d SMA AND SPY > 100d SMA; TLT otherwise.
     - Rebalance every 5 bars (weekly).
     - Leaderboard: IS Calmar 0.78, h1=0.63, h2=1.30 (improving!),
       loss_mode_corr_to_top5=0.50, corr_to_top5=0.41.

  B. qqq_vs_xlv_rotation (sonnet-3, gen_6)
     - Hold QQQ or XLV — whichever has higher 60d total return; 5-bar
       minimum hold.
     - Leaderboard: IS Calmar 0.67, h1=0.81, h2=0.63 (stable),
       loss_mode_corr_to_top5=0.37, corr_to_top5=0.61.
     - Always-equity (no bond rotation), so this leg is the equity-beta
       backbone of the ensemble.

  C. smallcap_leadership_rotation (sonnet-9, gen_6)
     - When IWM 20d return > SPY 20d return: QQQ 60% + IWM 37%.
     - Otherwise: SPY 60% + TLT 37%.
     - Rebalance every 5 bars (weekly).
     - Leaderboard: IS Calmar 0.61, h1=0.76, h2=0.58 (stable),
       loss_mode_corr_to_top5=0.28 (lowest of the trio), corr_to_top5=0.60.

Why this triplet:
  - Three structurally orthogonal signal types:
      A: credit trend (JNK 30d SMA) + equity trend (SPY 100d SMA)
      B: cross-sector momentum (QQQ vs XLV 60d return)
      C: cross-size momentum (IWM vs SPY 20d return)
  - All three components have h2 >= 0.58 (avoid the brief's flagged
    fragile components: risk_parity_4asset_vnq h2=0.33,
    qqq_vs_xlu_rotation h2=0.29).
  - Average loss_mode_corr_to_top5 = 0.38 — well below the 0.85 leaderboard
    leader cluster.
  - Average corr_to_top5 = 0.53.
  - Two components route to QQQ in risk-on regimes (A and C-bullish), one
    routes to XLV when healthcare leads (B). Sells happen in risk-off
    regimes (A->TLT, B switches to whichever ETF outperforms, C->SPY+TLT).

Composition rule: equal-weight (1/N).
  - Each component independently produces a target weight per symbol
    (capped at COMPONENT_WEIGHT = 1/3).
  - Symbols held by multiple components net naturally (e.g. if A wants
    QQQ 1.0 * 1/3 = 0.333, B wants QQQ 0.97 * 1/3 = 0.323, C wants
    QQQ 0.60 * 1/3 = 0.20, total target QQQ ~ 0.86).
  - Final exposure capped by EXPOSURE_CAP = 0.97 to leave a tiny cash
    buffer for slippage/rounding.

Hypothesis: combining 3 low-LMC stable-h2 strategies will produce a
higher Calmar than any one alone because their drawdowns occur at
different times (different signal regimes).

Pattern follows gen5_ensemble_bond_credit_seasonal.py.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — hy_credit_qqq_rotation parameters
# ---------------------------------------------------------------------------
A_JNK_MA = 30
A_SPY_MA = 100

# ---------------------------------------------------------------------------
# Component B — qqq_vs_xlv_rotation parameters
# ---------------------------------------------------------------------------
B_MOMENTUM_WINDOW = 60

# ---------------------------------------------------------------------------
# Component C — smallcap_leadership_rotation parameters
# ---------------------------------------------------------------------------
C_RS_WINDOW = 20

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97
ENSEMBLE_REBALANCE_EVERY = 5  # weekly
WARMUP_BARS = 110  # leave room for SPY 100d MA component
MIN_TRADE_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component A: hy_credit_qqq_rotation target weights (full 1.0)."""
    warmup = max(A_JNK_MA, A_SPY_MA) + 10
    if ctx.idx < warmup:
        return None

    jnk_bullish = False
    try:
        jnk_hist = ctx.history("JNK")
        if jnk_hist is not None and len(jnk_hist) >= A_JNK_MA + 1:
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= A_JNK_MA + 1:
                jnk_sma = float(jnk_close.iloc[-A_JNK_MA:].mean())
                jnk_bullish = float(jnk_close.iloc[-1]) > jnk_sma
    except KeyError:
        return None

    spy_bull = False
    try:
        spy_hist = ctx.history("SPY")
        if spy_hist is not None and len(spy_hist) >= A_SPY_MA + 1:
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= A_SPY_MA + 1:
                spy_sma = float(spy_close.iloc[-A_SPY_MA:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
    except KeyError:
        return None

    closes_now = ctx.closes()
    if jnk_bullish and spy_bull:
        return {"QQQ": 1.0} if "QQQ" in closes_now.index else None
    return {"TLT": 1.0} if "TLT" in closes_now.index else None


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component B: qqq_vs_xlv_rotation target weights (full 1.0)."""
    if ctx.idx < B_MOMENTUM_WINDOW + 5:
        return None

    try:
        qqq_hist = ctx.history("QQQ")
        xlv_hist = ctx.history("XLV")
    except KeyError:
        return None

    qqq_close = qqq_hist["close"].dropna() if qqq_hist is not None else None
    xlv_close = xlv_hist["close"].dropna() if xlv_hist is not None else None
    if qqq_close is None or xlv_close is None:
        return None
    if len(qqq_close) < B_MOMENTUM_WINDOW + 1 or len(xlv_close) < B_MOMENTUM_WINDOW + 1:
        return None

    qqq_start = float(qqq_close.iloc[-B_MOMENTUM_WINDOW - 1])
    xlv_start = float(xlv_close.iloc[-B_MOMENTUM_WINDOW - 1])
    if qqq_start <= 0 or xlv_start <= 0:
        return None

    qqq_mom = float(qqq_close.iloc[-1]) / qqq_start - 1.0
    xlv_mom = float(xlv_close.iloc[-1]) / xlv_start - 1.0
    if not (np.isfinite(qqq_mom) and np.isfinite(xlv_mom)):
        return None

    target_sym = "QQQ" if qqq_mom >= xlv_mom else "XLV"
    closes_now = ctx.closes()
    if target_sym not in closes_now.index:
        return None
    return {target_sym: 1.0}


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Component C: smallcap_leadership_rotation target weights."""
    if ctx.idx < C_RS_WINDOW + 5:
        return None

    try:
        iwm_hist = ctx.history("IWM")
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    iwm_close = iwm_hist["close"].dropna() if iwm_hist is not None else None
    spy_close = spy_hist["close"].dropna() if spy_hist is not None else None
    if iwm_close is None or spy_close is None:
        return None
    if len(iwm_close) < C_RS_WINDOW + 1 or len(spy_close) < C_RS_WINDOW + 1:
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


class EnsembleCreditSectorBreadth(Strategy):
    """Equal-weight ensemble of 3 low-LMC stable-h2 gen_6 survivors."""

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

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        a_w = _component_a_weights(ctx)
        b_w = _component_b_weights(ctx)
        c_w = _component_c_weights(ctx)

        target: dict[str, float] = {}
        for component_targets in (a_w, b_w, c_w):
            if component_targets is None:
                continue
            for sym, w in component_targets.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

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


NAME = "opus3_ensemble_credit_sector_breadth"
HYPOTHESIS = (
    "Equal-weight ensemble of 3 stable-h2 low-LMC gen6 survivors: "
    "hy_credit_qqq_rotation (JNK 30d MA + SPY 100d trend gating QQQ/TLT, "
    "LMC=0.50, h2>>h1), qqq_vs_xlv_rotation (60d momentum binary tech vs "
    "healthcare, LMC=0.37), smallcap_leadership_rotation (IWM 20d vs SPY "
    "20d return gating QQQ+IWM vs SPY+TLT, LMC=0.28); each component sized "
    "1/3, overlapping positions netted, total exposure capped 0.97; "
    "orthogonal signals (credit+trend, sector momentum, size factor)."
)
UNIVERSE = ["JNK", "SPY", "QQQ", "TLT", "XLV", "IWM"]

STRATEGY = EnsembleCreditSectorBreadth()
