"""opus-3 gen_9 ensemble #2 — Equal-weight triplet (vol-pct + EM + GDX/GLD).

Components (measured pairwise IS-return Pearson corrs via corr_dump):

    A. gen9_gdx_gld_momentum_sp500
       - IS Calmar 0.68, lmc 0.42
       - Risk-on -> top-15 SP500 63d-mom + 200d-SMA filter. Risk-off -> IEF.

    B. gen9_gen9_sp500_vol_pct_skipmon_momentum
       - IS Calmar 0.65, lmc 0.55
       - Calm (vol-pct <0.75) -> top-20 SP500 by 126d-21d skip-mom, inverse-vol
         weighted. Neutral -> SPY. Stress (vol-pct >0.90) -> TLT.

    C. gen9_em_dm_ratio_sp500_gate
       - IS Calmar 0.65
       - EM-leads-DM + SPY bull -> top-15 SP500 63d-mom + 200d-SMA filter.
         Otherwise TLT 60% + IEF 37%.

Measured pairwise corrs (corr_dump):
    A-B = +0.31  (different gates, both SP500 cross-sectional momentum upside)
    A-C = +0.31  (commodity gate vs cross-region gate)
    B-C = +0.17  (vol-percentile vs cross-region differential — fully orthogonal)
    mean |corr| = 0.26

Combining rule: EQUAL-WEIGHT, 1/3 per component, symbol-level netting,
EXPOSURE_CAP = 0.97. Identical structure to the curated ensemble
``ensemble_bond_credit_seasonal.py`` so the comparison is apples-to-apples.

Why two ensembles with different rules:
    - opus3_ensemble_gold_em_invvol (ensemble #1): inverse-vol weighted on
      Triplet 1 {gdx_gld, iau_gold, em_dm}; lower mean-corr (0.23) but a more
      complex weighting scheme that adds shadow-curve tracking state.
    - this (ensemble #2): equal-weight on Triplet 2 {gdx_gld, vol_pct, em_dm};
      simpler weighting and a different B-component (vol-pct gate rather than
      gold trend), which uses a SPY-realized-vol signal that's structurally
      different from gold and EM proxies.

Stability target: beat curated gen5_ensemble_bond_credit_seasonal's
OOS Calmar of 0.53 (78% retention) on stability, not gen8's IS 0.97
that collapsed to 0.31 OOS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# ---------------------------------------------------------------------------
# Component A — gdx_gld_momentum_sp500
# ---------------------------------------------------------------------------
A_MOMENTUM_WINDOW = 63
A_TREND_WINDOW = 200
A_SIGNAL_WINDOW = 42
A_TOP_K = 15

# ---------------------------------------------------------------------------
# Component B — sp500_vol_pct_skipmon_momentum
# ---------------------------------------------------------------------------
B_MOM_LOOKBACK = 126
B_MOM_SKIP = 21
B_VOL_WINDOW = 20
B_VOL_PCT_WINDOW = 252
B_CALM_THRESHOLD = 0.75
B_STRESS_THRESHOLD = 0.90
B_TOP_K = 20
B_INV_VOL_WINDOW = 20

# ---------------------------------------------------------------------------
# Component C — em_dm_ratio_sp500_gate
# ---------------------------------------------------------------------------
C_MOMENTUM_WINDOW = 63
C_TREND_WINDOW = 200
C_SIGNAL_WINDOW = 60
C_TOP_K = 15

# ---------------------------------------------------------------------------
# Ensemble params
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97
REBALANCE_EVERY = 10
WARMUP_BARS = max(
    A_MOMENTUM_WINDOW + A_TREND_WINDOW,
    B_MOM_LOOKBACK + B_MOM_SKIP + B_VOL_PCT_WINDOW,
    C_MOMENTUM_WINDOW + C_TREND_WINDOW,
) + 15

_NON_RANK_SYMS = {
    "SPY", "TLT", "IEF", "LQD", "GDX", "GLD", "IAU", "VWO", "VEA",
}


# ===========================================================================
# Component A (same as ensemble #1)
# ===========================================================================
def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = max(A_TREND_WINDOW, A_SIGNAL_WINDOW) + 5
    if ctx.idx < warmup:
        return None
    closes = ctx.closes()
    if closes.empty:
        return None
    live_all = {s: float(p) for s, p in closes.items() if float(p) > 0}

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < A_TREND_WINDOW:
        return None
    spy_sma = float(spy_hist["close"].iloc[-A_TREND_WINDOW:].mean())
    spy_price = live_all.get("SPY", 0.0)
    spy_bull = spy_price > 0 and spy_price > spy_sma

    if not spy_bull:
        return {"IEF": 1.0} if "IEF" in live_all else None

    try:
        gdx_hist = ctx.history("GDX")
        gld_hist = ctx.history("GLD")
    except KeyError:
        return {"IEF": 1.0} if "IEF" in live_all else None
    if len(gdx_hist) < A_SIGNAL_WINDOW + 1 or len(gld_hist) < A_SIGNAL_WINDOW + 1:
        return {"IEF": 1.0} if "IEF" in live_all else None

    gdx_close = gdx_hist["close"]
    gld_close = gld_hist["close"]
    gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-A_SIGNAL_WINDOW] - 1.0)
    gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-A_SIGNAL_WINDOW] - 1.0)
    if not (np.isfinite(gdx_ret) and np.isfinite(gld_ret)):
        return {"IEF": 1.0} if "IEF" in live_all else None
    if not (gdx_ret > gld_ret):
        return {"IEF": 1.0} if "IEF" in live_all else None

    prices = ctx.closes_window(A_MOMENTUM_WINDOW + 5)
    if len(prices) < A_MOMENTUM_WINDOW:
        return {"IEF": 1.0} if "IEF" in live_all else None

    scores: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < A_MOMENTUM_WINDOW:
            continue
        p_start = float(col.iloc[-A_MOMENTUM_WINDOW])
        if p_start <= 0:
            continue
        r = float(col.iloc[-1] / p_start - 1.0)
        if np.isfinite(r):
            scores[sym] = r

    if len(scores) < A_TOP_K:
        return {"IEF": 1.0} if "IEF" in live_all else None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: list[str] = []
    for sym, _ in ranked:
        if len(selected) >= A_TOP_K:
            break
        hist = ctx.history(sym)
        if len(hist) < A_TREND_WINDOW:
            continue
        sma = float(hist["close"].iloc[-A_TREND_WINDOW:].mean())
        price = live_all.get(sym, 0.0)
        if price > sma:
            selected.append(sym)

    if not selected:
        return {"IEF": 1.0} if "IEF" in live_all else None
    per_w = 1.0 / len(selected)
    return {sym: per_w for sym in selected}


# ===========================================================================
# Component B — vol-pct gate skip-month momentum
# ===========================================================================
def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = B_MOM_LOOKBACK + B_MOM_SKIP + B_VOL_PCT_WINDOW + 10
    if ctx.idx < warmup:
        return None

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < B_VOL_PCT_WINDOW + B_VOL_WINDOW + 5:
        return None

    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < B_VOL_PCT_WINDOW + B_VOL_WINDOW + 2:
        return None
    spy_logret = np.log(spy_close.values[1:] / spy_close.values[:-1])
    if len(spy_logret) < B_VOL_WINDOW + B_VOL_PCT_WINDOW:
        return None

    rolling_vols: list[float] = []
    for i in range(B_VOL_PCT_WINDOW):
        end = len(spy_logret) - i
        start = end - B_VOL_WINDOW
        if start < 0:
            break
        rolling_vols.append(float(np.std(spy_logret[start:end])))

    if len(rolling_vols) < 10:
        return None

    current_vol = rolling_vols[0]
    vol_pct = float(np.mean([v < current_vol for v in rolling_vols[1:]]))

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

    if vol_pct > B_STRESS_THRESHOLD:
        return {"TLT": 1.0} if "TLT" in live else None

    if vol_pct >= B_CALM_THRESHOLD:
        return {"SPY": 1.0} if "SPY" in live else None

    # Calm regime: top-K skip-month, inverse-vol weighted
    need = B_MOM_LOOKBACK + B_MOM_SKIP + 2
    prices = ctx.closes_window(need)
    if len(prices) < need - 1:
        return {"SPY": 1.0} if "SPY" in live else None

    scores: dict[str, float] = {}
    inv_vols: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < B_MOM_LOOKBACK + B_MOM_SKIP:
            continue
        p_end = float(col.iloc[-B_MOM_SKIP - 1])
        p_start = float(col.iloc[-(B_MOM_LOOKBACK + B_MOM_SKIP)])
        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
            continue
        ret = p_end / p_start - 1.0
        tail = col.iloc[-B_INV_VOL_WINDOW - 1:]
        if len(tail) < B_INV_VOL_WINDOW + 1:
            continue
        logr = np.log(tail.values[1:] / tail.values[:-1])
        rv = float(np.std(logr))
        if rv <= 1e-6 or not np.isfinite(rv):
            continue
        scores[sym] = ret
        inv_vols[sym] = 1.0 / rv

    if len(scores) < B_TOP_K:
        return {"SPY": 1.0} if "SPY" in live else None

    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:B_TOP_K]
    iv_sum = sum(inv_vols[s] for s in ranked)
    if iv_sum <= 0:
        return {"SPY": 1.0} if "SPY" in live else None
    return {sym: inv_vols[sym] / iv_sum for sym in ranked if sym in live}


# ===========================================================================
# Component C (same as ensemble #1)
# ===========================================================================
def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    warmup = max(C_TREND_WINDOW, C_SIGNAL_WINDOW) + 5
    if ctx.idx < warmup:
        return None
    closes = ctx.closes()
    if closes.empty:
        return None
    live_all = {s: float(p) for s, p in closes.items() if float(p) > 0}

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < C_TREND_WINDOW:
        return None
    spy_sma = float(spy_hist["close"].iloc[-C_TREND_WINDOW:].mean())
    spy_price = live_all.get("SPY", 0.0)
    spy_bull = spy_price > 0 and spy_price > spy_sma

    if not spy_bull:
        return {"TLT": 0.619, "IEF": 0.381} if ("TLT" in live_all and "IEF" in live_all) else None

    try:
        vwo_hist = ctx.history("VWO")
        vea_hist = ctx.history("VEA")
    except KeyError:
        return {"TLT": 0.619, "IEF": 0.381} if ("TLT" in live_all and "IEF" in live_all) else None

    em_risk_on = False
    if len(vwo_hist) >= C_SIGNAL_WINDOW + 1 and len(vea_hist) >= C_SIGNAL_WINDOW + 1:
        vwo_close = vwo_hist["close"]
        vea_close = vea_hist["close"]
        vwo_ret = float(vwo_close.iloc[-1] / vwo_close.iloc[-C_SIGNAL_WINDOW] - 1.0)
        vea_ret = float(vea_close.iloc[-1] / vea_close.iloc[-C_SIGNAL_WINDOW] - 1.0)
        if np.isfinite(vwo_ret) and np.isfinite(vea_ret):
            em_risk_on = vwo_ret > vea_ret

    if not em_risk_on:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    prices = ctx.closes_window(C_MOMENTUM_WINDOW + 5)
    if len(prices) < C_MOMENTUM_WINDOW:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    scores: dict[str, float] = {}
    for sym in prices.columns:
        if sym in _NON_RANK_SYMS:
            continue
        col = prices[sym].dropna()
        if len(col) < C_MOMENTUM_WINDOW:
            continue
        p_start = float(col.iloc[-C_MOMENTUM_WINDOW])
        if p_start <= 0:
            continue
        r = float(col.iloc[-1] / p_start - 1.0)
        if np.isfinite(r):
            scores[sym] = r

    if len(scores) < C_TOP_K:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected: list[str] = []
    for sym, _ in ranked:
        if len(selected) >= C_TOP_K:
            break
        hist = ctx.history(sym)
        if len(hist) < C_TREND_WINDOW:
            continue
        sma = float(hist["close"].iloc[-C_TREND_WINDOW:].mean())
        price = live_all.get(sym, 0.0)
        if price > sma:
            selected.append(sym)

    if not selected:
        if "TLT" in live_all and "IEF" in live_all:
            return {"TLT": 0.619, "IEF": 0.381}
        return None

    per_w = 1.0 / len(selected)
    return {sym: per_w for sym in selected}


# ===========================================================================
# Ensemble strategy
# ===========================================================================
class Opus3EnsembleVolPctEmGoldEqWt(Strategy):
    """Equal-weight (1/3 each, netted, capped at 0.97) ensemble of 3 low-corr
    survivors (opus-3, gen_9, ensemble #2). Pure equal-weight composition —
    no inverse-vol, no regime gate, no voting — to isolate the diversification
    benefit from any active weighting scheme.
    """

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
        for comp_target in (a_w, b_w, c_w):
            if comp_target is None:
                continue
            for sym, w in comp_target.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

        total = sum(target.values())
        if total <= 0:
            return []
        if total > self.exposure_cap:
            scale = self.exposure_cap / total
            target = {k: v * scale for k, v in target.items()}

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
            n = int(equity * weight / price)
            if n > 0:
                target_shares[sym] = n

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -1:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > 1:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))
        return orders


def _universe() -> list[str]:
    extra = ["SPY", "TLT", "IEF", "LQD", "GDX", "GLD", "VWO", "VEA"]
    return sp500_tickers() + extra


NAME = "opus3_ensemble_volpct_em_gold_eqwt"
HYPOTHESIS = (
    "Equal-weight (1/3 each, netted, cap 0.97) triplet ensemble: "
    "A=gdx_gld_momentum_sp500 (GDX-vs-GLD 42d return gate), "
    "B=sp500_vol_pct_skipmon_momentum (SPY 20d vol-percentile gate, 126d-21d skip-mom), "
    "C=em_dm_ratio_sp500_gate (VWO-vs-VEA 60d return gate); measured pairwise corrs "
    "A-B=0.31, A-C=0.31, B-C=0.17 via corr_dump; biweekly rebalance."
)
UNIVERSE = _universe

STRATEGY = Opus3EnsembleVolPctEmGoldEqWt()
