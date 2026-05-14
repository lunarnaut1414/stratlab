"""Equal-weight ensemble of 3 orthogonal-loss-mode gen_7 survivors.

Components (each sized 1/3 of capital, target weights summed and netted):

  A. gld_gdx_regime_momentum (sonnet, gen_7)
     - Risk-on regime: GDX 20d return > GLD 20d return AND SPY > 200d SMA.
       In risk-on, hold top-10 SP500 stocks by 63d momentum.
     - Risk-off (gold leads miners OR bear market): GLD 60% + TLT 37%.
     - Rebalance every 10 bars.
     - Leaderboard: IS Calmar 0.86, h1=0.86, h2=?, loss_mode_corr_to_top5=0.42
       (LOWEST of the trio — distinct failure mode from SP500-momentum cluster).

  B. jnk_iwm_dual_gate_qqq (sonnet-7, gen_7)
     - Bull (SPY>200d SMA) AND credit_bull (JNK>50d SMA) AND
       smallcap_bull (IWM 20d return > 0): hold QQQ.
     - Bull AND credit_bull only: hold SPY.
     - Otherwise: SHY 50% + TLT 47%.
     - Rebalance every 5 bars (weekly).
     - Leaderboard: IS Calmar 0.58, h1=0.33, h2=1.37, loss_mode_corr_to_top5=0.55.
     - h2-dominant (improving stability), distinct credit+breadth signal class.

  C. realized_vol_carry_spy (sonnet, gen_7)
     - SPY 21d realized vol vs 33rd/67th percentile of 90d RV distribution:
       calm (RV <= p33) -> SPY 90%
       middle (p33 < RV < p67) -> SPY 70%
       stressed (RV >= p67) -> SPY 50% + TLT 47%
     - Rebalance every 5 bars (weekly).
     - Leaderboard: IS Calmar 1.04 (HIGHEST of trio), max_dd -9.2% (smallest of trio).
     - Pure realized-vol signal — orthogonal to credit/RS-momentum signals.

Why this triplet:
  - Three structurally orthogonal signal types:
      A: cross-commodity relative strength (GDX vs GLD 20d return) +
         cross-sectional SP500 stock momentum
      B: credit trend (JNK 50d SMA) + small-cap absolute return (IWM 20d) +
         broad equity trend (SPY 200d SMA)
      C: realized volatility regime (SPY 21d RV vs its own 90d distribution)
  - The brief explicitly identifies these as the "three orthogonal-loss-mode
    candidates": loss_mode_corr_to_top5 = 0.42 (A) and 0.55 (B) are the
    LOWEST in the gen_7 leaderboard, indicating they break differently than
    the dominant SP500-momentum cluster.
  - Asset routing differs across regimes:
      A risk-on -> SP500 stocks; risk-off -> GLD+TLT
      B risk-on -> QQQ or SPY; risk-off -> SHY+TLT
      C calm -> SPY 90%; stressed -> SPY 50% + TLT 47%
    Overlap on SPY/TLT lets the ensemble net positions naturally; the
    differences on QQQ vs GLD vs SP500-stocks ensure each component
    contributes a distinct exposure when its signal fires.

Composition rule: equal-weight (1/N).
  - Each component independently produces a target weight per symbol
    (capped at COMPONENT_WEIGHT = 1/3).
  - Symbols held by multiple components net naturally.
  - Final exposure capped by EXPOSURE_CAP = 0.97 to leave a small cash
    buffer for slippage/rounding.

Rebalance frequency: weekly (every 5 bars). Component A's native cadence is
10 bars but checking weekly only causes A's signal to update every other
ensemble step, which is fine since A's regime persists for weeks/months.

Hypothesis: combining 3 orthogonal-loss-mode strategies will produce a
higher Calmar than any one alone because their drawdowns occur in
different regimes:
  - A breaks when commodity rotation is wrong (gold-miner divergence)
  - B breaks when credit/breadth disagrees with equity (e.g. JNK rolling
    over while SPY holds)
  - C breaks when vol regimes are choppy (whipsaw across percentile)

Pattern follows gen5_ensemble_bond_credit_seasonal.py and
gen6_opus3_ensemble_credit_sector_breadth.py.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — gld_gdx_regime_momentum parameters
# ---------------------------------------------------------------------------
A_MOMENTUM_WINDOW = 20    # GDX vs GLD 20d return
A_STOCK_MOM_WINDOW = 63   # SP500 cross-section 63d momentum
A_TOP_K = 10
A_TREND_WINDOW = 200      # SPY 200d SMA secondary gate

# ---------------------------------------------------------------------------
# Component B — jnk_iwm_dual_gate_qqq parameters
# ---------------------------------------------------------------------------
B_JNK_MA = 50             # JNK 50d SMA
B_IWM_WINDOW = 20         # IWM 20d absolute return
B_TREND_WINDOW = 200

# ---------------------------------------------------------------------------
# Component C — realized_vol_carry_spy parameters
# ---------------------------------------------------------------------------
C_RV_WINDOW = 21          # 21d realized vol
C_MEDIAN_WINDOW = 90      # 90d window for percentile comparison
C_EXPOSURE_HIGH = 0.90
C_EXPOSURE_MID = 0.70
C_EXPOSURE_LOW = 0.50
C_TLT_DEFENSIVE = 0.47    # TLT add when stressed

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0   # each component sized at 1/3 of capital
EXPOSURE_CAP = 0.97             # total invested cap (leave cash buffer)
ENSEMBLE_REBALANCE_EVERY = 5    # weekly
WARMUP_BARS = 220               # leave room for slowest component (SPY 200d MA + buffer)
MIN_TRADE_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """A: GLD/GDX RS regime gating SP500 momentum vs GLD+TLT."""
    warmup = max(A_TREND_WINDOW, A_STOCK_MOM_WINDOW) + 10
    if ctx.idx < warmup:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None

    # SPY 200d SMA secondary gate
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

    # GDX vs GLD 20d return (primary regime)
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
        # Risk-off: GLD 60% + TLT 37% (component-internal)
        if "GLD" in closes_now.index:
            weights["GLD"] = 0.60
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.37
    else:
        # Risk-on: top-K SP500 stocks by 63d momentum
        prices = ctx.closes_window(A_STOCK_MOM_WINDOW + 5)
        if len(prices) < A_STOCK_MOM_WINDOW:
            # fallback to SPY if not enough cross-sectional data
            if "SPY" in closes_now.index:
                weights["SPY"] = 0.97
        else:
            scores: dict[str, float] = {}
            # exclude allocator ETFs from the cross-sectional ranking
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

    # SPY 200d outer gate
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
        # Defensive: SHY 50% + TLT 47%
        if "SHY" in closes_now.index:
            weights["SHY"] = 0.50
        if "TLT" in closes_now.index:
            weights["TLT"] = 0.47
        return weights if weights else None

    # JNK 50d SMA (credit signal)
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

    # IWM 20d positive (small-cap signal)
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
    """C: SPY 21d RV vs 33rd/67th pct of 90d distribution -> SPY 90/70/50 + TLT."""
    warmup = C_MEDIAN_WINDOW + C_RV_WINDOW + 10
    if ctx.idx < warmup:
        return None

    closes_now = ctx.closes()
    if closes_now.empty:
        return None
    if "SPY" not in closes_now.index:
        return None

    # default: middle regime
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

        # Current 21d RV (annualized)
        current_rv = float(np.std(log_rets[-C_RV_WINDOW:]) * np.sqrt(252))

        # Rolling 21d RV across the last C_MEDIAN_WINDOW days
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


class EnsembleOrthogonalLossModes(Strategy):
    """Equal-weight ensemble of 3 orthogonal-loss-mode gen_7 survivors."""

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

        # Aggregate with equal-weight scaling.
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

        if not target:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Build target share counts
        target_shares: dict[str, int] = {}
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            shares = int(equity * weight / price)
            if shares > 0:
                target_shares[sym] = shares

        orders: list[Order] = []

        # 1. Liquidate positions not in target (sells first to free cash)
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # 2. Reduce overweights vs target
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
    """Universe = SP500 (for component A momentum) plus all ETFs each
    component routes to: SPY, TLT, GLD, GDX (A), QQQ, IWM, SHY, JNK (B), SPY, TLT (C)."""
    from stratlab.data.universe import sp500_tickers
    extra = ["SPY", "TLT", "GLD", "GDX", "QQQ", "IWM", "SHY", "JNK"]
    return sp500_tickers() + extra


NAME = "opus3_ensemble_orthogonal_loss_modes"
HYPOTHESIS = (
    "Equal-weight ensemble of 3 orthogonal-loss-mode gen_7 survivors: "
    "A=gld_gdx_regime_momentum (GDX/GLD 20d RS gating SP500-mom vs GLD+TLT, lmc=0.42), "
    "B=jnk_iwm_dual_gate_qqq (JNK 50d MA + IWM 20d positive return tri-state QQQ/SPY/SHY+TLT, lmc=0.55), "
    "C=realized_vol_carry_spy (SPY 21d RV vs 33rd/67th pct of 90d distribution gating SPY 90/70/50 + TLT); "
    "each component sized 1/3, overlapping positions netted, total exposure capped 0.97; weekly rebalance"
)
UNIVERSE = _universe

STRATEGY = EnsembleOrthogonalLossModes()
