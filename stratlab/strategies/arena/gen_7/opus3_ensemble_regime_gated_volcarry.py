"""Regime-gated ensemble of the same 3 orthogonal-loss-mode gen_7 survivors.

Variant of opus3_ensemble_orthogonal_loss_modes.py with a different
combining rule: instead of always-on equal-weighting of A, B, C, this
strategy uses SPY 200d SMA as a regime gate to switch between
"offensive" and "defensive" composition.

Combining rule:
  - Bull (SPY > 200d SMA): equal-weight all three components 1/3 each.
    Each component's internal logic still fires (e.g. component A may
    decide risk-off internally if GDX < GLD).
  - Bear (SPY <= 200d SMA): hold ONLY component C (realized_vol_carry_spy)
    — this is the lowest-max-dd component (-9.2% vs A's -17.4% and B's
    h1=0.33 fragile half) and has built-in vol-regime defensive sizing
    (drops SPY exposure to 50% + adds 47% TLT when realized vol is in the
    top tercile of its own 90d distribution).

Components (same as opus3_ensemble_orthogonal_loss_modes — see that file
for full description):

  A. gld_gdx_regime_momentum   (lmc 0.42)
  B. jnk_iwm_dual_gate_qqq     (lmc 0.55)
  C. realized_vol_carry_spy    (highest IS Calmar 1.04, smallest DD)

Why regime-gate to C only in bear markets:
  - In SPY > 200d SMA the bull-market gates of A and B already filter
    out most defensive periods; equal-weighting in bull makes sense.
  - In SPY <= 200d SMA, components A and B both retreat to bond defensive
    baskets that overlap heavily (TLT + GLD/SHY); aggregating these adds
    no diversification, just bond-duration concentration.
  - Component C uses a continuous vol regime that is independent of SPY's
    trend and naturally sizes down equity in stressed regimes; letting C
    carry the bear leg keeps drawdown shallow without forcing 100% bonds.
  - 2010-2018 IS has VIX < 18 on 68% of days (calm-biased per phase2_brief);
    SPY is below 200d SMA only ~12-15% of bars; so the defensive arm is
    a small fraction of capital exposure even when triggered.

Hypothesis: combining a bull-regime equal-weight ensemble with a
defensive-only fallback to the smallest-DD component will produce a
higher Calmar than the always-on equal-weight version, because the bear
leg (where A and B's bond baskets are redundant) is replaced by C's
finer-grained vol regime allocation.

Pattern follows gen5_ensemble_bond_credit_seasonal.py and
gen6_opus3_ensemble_credit_sector_breadth.py with regime-gating added.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — gld_gdx_regime_momentum parameters
# ---------------------------------------------------------------------------
A_MOMENTUM_WINDOW = 20
A_STOCK_MOM_WINDOW = 63
A_TOP_K = 10
A_TREND_WINDOW = 200

# ---------------------------------------------------------------------------
# Component B — jnk_iwm_dual_gate_qqq parameters
# ---------------------------------------------------------------------------
B_JNK_MA = 50
B_IWM_WINDOW = 20
B_TREND_WINDOW = 200

# ---------------------------------------------------------------------------
# Component C — realized_vol_carry_spy parameters
# ---------------------------------------------------------------------------
C_RV_WINDOW = 21
C_MEDIAN_WINDOW = 90
C_EXPOSURE_HIGH = 0.90
C_EXPOSURE_MID = 0.70
C_EXPOSURE_LOW = 0.50
C_TLT_DEFENSIVE = 0.47

# ---------------------------------------------------------------------------
# Regime gate parameters
# ---------------------------------------------------------------------------
REGIME_TREND_WINDOW = 200       # SPY 200d SMA = bull/bear switch

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT_BULL = 1.0 / 3.0   # bull mode: equal-weight 3 components
EXPOSURE_CAP = 0.97
ENSEMBLE_REBALANCE_EVERY = 5         # weekly
WARMUP_BARS = 220
MIN_TRADE_DELTA = 1


def _spy_bull(ctx: BarContext) -> bool | None:
    """SPY 200d SMA regime gate. None if not enough data."""
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if spy_hist is None or len(spy_hist) < REGIME_TREND_WINDOW:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < REGIME_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-REGIME_TREND_WINDOW:].mean())
    return float(spy_close.iloc[-1]) > spy_sma


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """A: GLD/GDX RS regime gating SP500 momentum vs GLD+TLT."""
    warmup = max(A_TREND_WINDOW, A_STOCK_MOM_WINDOW) + 10
    if ctx.idx < warmup:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    bull_market = True
    try:
        spy_hist = ctx.history("SPY")
        if spy_hist is not None and len(spy_hist) >= A_TREND_WINDOW:
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= A_TREND_WINDOW:
                spy_sma = float(spy_close.iloc[-A_TREND_WINDOW:].mean())
                bull_market = float(spy_close.iloc[-1]) > spy_sma
    except KeyError:
        return None

    risk_on = True
    try:
        gld_hist = ctx.history("GLD")
        gdx_hist = ctx.history("GDX")
        if (gld_hist is not None and gdx_hist is not None
                and len(gld_hist) >= A_MOMENTUM_WINDOW + 1
                and len(gdx_hist) >= A_MOMENTUM_WINDOW + 1):
            gld_close = gld_hist["close"].dropna()
            gdx_close = gdx_hist["close"].dropna()
            if (len(gld_close) >= A_MOMENTUM_WINDOW + 1
                    and len(gdx_close) >= A_MOMENTUM_WINDOW + 1):
                gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-A_MOMENTUM_WINDOW - 1] - 1.0)
                gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-A_MOMENTUM_WINDOW - 1] - 1.0)
                risk_on = (gdx_ret > gld_ret)
    except KeyError:
        return None

    weights: dict[str, float] = {}
    if not bull_market or not risk_on:
        if "GLD" in closes_now.index:
            weights["GLD"] = 0.60
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.37
    else:
        prices = ctx.closes_window(A_STOCK_MOM_WINDOW + 5)
        if len(prices) < A_STOCK_MOM_WINDOW:
            if "SPY" in closes_now.index:
                weights["SPY"] = 0.97
        else:
            scores: dict[str, float] = {}
            etf_blacklist = {
                "SPY", "TLT", "GLD", "GDX", "QQQ", "IWM", "JNK",
                "LQD", "HYG", "SHY", "IEF",
            }
            for sym in prices.columns:
                if sym in etf_blacklist:
                    continue
                col = prices[sym].dropna()
                if len(col) < A_STOCK_MOM_WINDOW:
                    continue
                start_p = float(col.iloc[-A_STOCK_MOM_WINDOW])
                if start_p <= 0:
                    continue
                ret = float(col.iloc[-1] / start_p - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < A_TOP_K:
                if "SPY" in closes_now.index:
                    weights["SPY"] = 0.97
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                longs = ranked[:A_TOP_K]
                per_w = 0.97 / len(longs)
                for sym in longs:
                    weights[sym] = per_w

    return weights if weights else None


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """B: JNK 50d MA + IWM 20d positive return tri-state QQQ/SPY/SHY+TLT."""
    warmup = max(B_JNK_MA, B_TREND_WINDOW, B_IWM_WINDOW) + 10
    if ctx.idx < warmup:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    if len(spy_hist) < B_TREND_WINDOW:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < B_TREND_WINDOW:
        return None
    spy_sma = float(spy_close.iloc[-B_TREND_WINDOW:].mean())
    bull = float(spy_close.iloc[-1]) > spy_sma

    weights: dict[str, float] = {}

    if not bull:
        if "SHY" in closes_now.index:
            weights["SHY"] = 0.50
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.47
        return weights if weights else None

    credit_bull = False
    try:
        jnk_hist = ctx.history("JNK")
        if jnk_hist is not None and len(jnk_hist) >= B_JNK_MA + 1:
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= B_JNK_MA + 1:
                jnk_sma = float(jnk_close.iloc[-B_JNK_MA:].mean())
                credit_bull = float(jnk_close.iloc[-1]) > jnk_sma
    except KeyError:
        pass

    smallcap_bull = False
    try:
        iwm_hist = ctx.history("IWM")
        if iwm_hist is not None and len(iwm_hist) >= B_IWM_WINDOW + 1:
            iwm_close = iwm_hist["close"].dropna()
            if len(iwm_close) >= B_IWM_WINDOW + 1:
                iwm_start = float(iwm_close.iloc[-B_IWM_WINDOW - 1])
                if iwm_start > 0:
                    iwm_ret = float(iwm_close.iloc[-1] / iwm_start - 1.0)
                    smallcap_bull = np.isfinite(iwm_ret) and iwm_ret > 0
    except KeyError:
        pass

    if credit_bull and smallcap_bull:
        if "QQQ" in closes_now.index:
            weights["QQQ"] = 0.97
    elif credit_bull:
        if "SPY" in closes_now.index:
            weights["SPY"] = 0.97
    else:
        if "SHY" in closes_now.index:
            weights["SHY"] = 0.50
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.47

    return weights if weights else None


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """C: SPY 21d RV vs 33rd/67th pct of 90d distribution."""
    warmup = C_MEDIAN_WINDOW + C_RV_WINDOW + 10
    if ctx.idx < warmup:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    if "SPY" not in closes_now.index:
        return None

    spy_w = C_EXPOSURE_MID
    tlt_w = 0.0

    try:
        spy_hist = ctx.history("SPY")
        if spy_hist is None or len(spy_hist) < C_MEDIAN_WINDOW + C_RV_WINDOW:
            return None
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < C_MEDIAN_WINDOW + C_RV_WINDOW:
            return None

        log_rets = np.log(spy_close.values[1:] / spy_close.values[:-1])
        if len(log_rets) < C_RV_WINDOW + C_MEDIAN_WINDOW:
            return None

        current_rv = float(np.std(log_rets[-C_RV_WINDOW:]) * np.sqrt(252))

        rv_series = []
        for i in range(C_MEDIAN_WINDOW):
            end_i = len(log_rets) - i
            start_i = end_i - C_RV_WINDOW
            if start_i < 0:
                break
            rv_series.append(float(np.std(log_rets[start_i:end_i]) * np.sqrt(252)))

        if not rv_series or not np.isfinite(current_rv):
            return None

        p33 = float(np.percentile(rv_series, 33))
        p67 = float(np.percentile(rv_series, 67))

        if current_rv <= p33:
            spy_w = C_EXPOSURE_HIGH
            tlt_w = 0.0
        elif current_rv >= p67:
            spy_w = C_EXPOSURE_LOW
            tlt_w = C_TLT_DEFENSIVE
        else:
            spy_w = C_EXPOSURE_MID
            tlt_w = 0.0
    except KeyError:
        return None

    weights: dict[str, float] = {"SPY": spy_w}
    if tlt_w > 0 and "TLT" in closes_now.index:
        weights["TLT"] = tlt_w
    return weights


class EnsembleRegimeGatedVolCarry(Strategy):
    """Regime-gated ensemble: bull -> 1/3 each A,B,C; bear -> only C."""

    def __init__(
        self,
        rebalance_every: int = ENSEMBLE_REBALANCE_EVERY,
        component_weight_bull: float = COMPONENT_WEIGHT_BULL,
        exposure_cap: float = EXPOSURE_CAP,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            component_weight_bull=component_weight_bull,
            exposure_cap=exposure_cap,
        )
        self.rebalance_every = int(rebalance_every)
        self.component_weight_bull = float(component_weight_bull)
        self.exposure_cap = float(exposure_cap)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < WARMUP_BARS:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        bull = _spy_bull(ctx)
        if bull is None:
            return []

        target: dict[str, float] = {}

        if bull:
            # Offensive: equal-weight all three
            a_w = _component_a_weights(ctx)
            b_w = _component_b_weights(ctx)
            c_w = _component_c_weights(ctx)
            for component_targets in (a_w, b_w, c_w):
                if component_targets is None:
                    continue
                for sym, w in component_targets.items():
                    target[sym] = target.get(sym, 0.0) + w * self.component_weight_bull
        else:
            # Defensive: only C, full 1.0 weight (no scaling by 1/3)
            c_w = _component_c_weights(ctx)
            if c_w is not None:
                for sym, w in c_w.items():
                    target[sym] = target.get(sym, 0.0) + w
            else:
                # C unavailable in bear (shouldn't happen post-warmup) — sit in cash
                pass

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    extra = ["SPY", "TLT", "GLD", "GDX", "QQQ", "IWM", "SHY", "JNK"]
    return sp500_tickers() + extra


NAME = "opus3_ensemble_regime_gated_volcarry"
HYPOTHESIS = (
    "Regime-gated ensemble of same 3 orthogonal-loss-mode gen7 survivors "
    "(gld_gdx_regime_momentum, jnk_iwm_dual_gate_qqq, realized_vol_carry_spy): "
    "when SPY>200d SMA equal-weight all three components 1/3 each (offensive); "
    "when SPY<=200d SMA hold ONLY component C (realized_vol_carry_spy) the "
    "lowest-max-dd component with native vol-regime defensive sizing; total "
    "exposure capped 0.97; weekly rebalance"
)
UNIVERSE = _universe

STRATEGY = EnsembleRegimeGatedVolCarry()
