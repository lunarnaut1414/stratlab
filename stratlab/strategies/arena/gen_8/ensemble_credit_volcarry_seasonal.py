"""Ensemble of Credit Regime + Vol Carry + Seasonal — gen_8 sonnet-2

Equal-weight ensemble of 3 structurally orthogonal diversifiers:

  A. JNK Credit Gate: QQQ 97% when JNK above 30d SMA (credit risk-on);
     TLT 97% when JNK below 30d SMA. Weekly signal. Pure credit regime.

  B. SPY Realized-Vol Carry: SPY 87% when 21d RV below 33rd pct of 90d
     distribution (calm); SPY 65%+TLT 32% when above 67th pct (stressed);
     SPY 75% in middle. Weekly. Pure vol-regime signal.

  C. Halloween Seasonal: SPY 95% Nov-Apr (winter risk-on); TLT 95% May-Oct.
     Calendar-only signal, orthogonal to market data.

Three signals are structurally orthogonal:
  - A fires off JNK credit spreads (fixed income credit)
  - B fires off SPY realized volatility (equity market vol)
  - C fires off calendar month (no market data at all)

Composition: each component sized at 1/3 of capital. Overlapping ETF
positions net (e.g. if A and C both want TLT, their TLT weights add).
Total exposure capped at 0.97.

Why this ensemble is distinct from existing leaderboard:
  - gen5_ensemble_bond_credit_seasonal uses SP500 stock selection in component A
    (correlated to xsect-momentum cluster); this uses JNK→QQQ (ETF only)
  - Different component B: vol carry (not credit spread crossover)
  - Orthogonal signals should reduce loss_mode_corr vs SP500 cluster

Rebalance: every 5 bars (weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Component A — JNK credit gate → QQQ/TLT
# ---------------------------------------------------------------------------
A_REBALANCE = 5
A_JNK_MA = 30

# ---------------------------------------------------------------------------
# Component B — SPY realized-vol carry → SPY/TLT
# ---------------------------------------------------------------------------
B_REBALANCE = 5
B_RV_WINDOW = 21
B_DIST_WINDOW = 90
B_EXPOSURE_CALM = 0.87     # below 33rd pct: SPY
B_EXPOSURE_MID = 0.75      # middle: SPY
B_EXPOSURE_STRESSED = 0.65 # above 67th pct: SPY
B_TLT_STRESSED = 0.32      # TLT complement in stressed

# ---------------------------------------------------------------------------
# Component C — Halloween seasonal → SPY/TLT
# ---------------------------------------------------------------------------
C_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}  # Nov-Apr: SPY
C_EXPOSURE = 0.95

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97
REBALANCE_EVERY = 5
WARMUP_BARS = 120  # accommodate vol distribution window
MIN_DELTA = 1


def _component_a_weights(ctx: BarContext) -> dict[str, float] | None:
    """JNK credit gate: QQQ when JNK > 30d SMA, TLT otherwise."""
    if ctx.idx < A_JNK_MA + 10:
        return None
    try:
        jnk_hist = ctx.history("JNK")
    except KeyError:
        return None
    jnk_close = jnk_hist["close"].dropna()
    if len(jnk_close) < A_JNK_MA:
        return None
    jnk_ma = float(jnk_close.iloc[-A_JNK_MA:].mean())
    jnk_now = float(jnk_close.iloc[-1])
    if jnk_now > jnk_ma:
        return {"QQQ": 1.0}
    else:
        return {"TLT": 1.0}


def _component_b_weights(ctx: BarContext) -> dict[str, float] | None:
    """SPY realized-vol carry: SPY/TLT allocation tiered by RV percentile."""
    if ctx.idx < B_DIST_WINDOW + B_RV_WINDOW + 10:
        return None
    try:
        spy_hist = ctx.history("SPY")
    except KeyError:
        return None
    spy_close = spy_hist["close"].dropna()
    if len(spy_close) < B_DIST_WINDOW + B_RV_WINDOW + 2:
        return None

    log_rets = np.log(spy_close.values[1:] / spy_close.values[:-1])

    # Current 21d realized vol
    if len(log_rets) < B_RV_WINDOW:
        return None
    current_rv = float(np.std(log_rets[-B_RV_WINDOW:]))
    if not np.isfinite(current_rv) or current_rv <= 0:
        return None

    # Rolling 21d RV distribution over last 90 days
    rv_series = []
    for i in range(B_DIST_WINDOW):
        end_i = len(log_rets) - i
        start_i = end_i - B_RV_WINDOW
        if start_i < 0:
            break
        rv_i = float(np.std(log_rets[start_i:end_i]))
        if np.isfinite(rv_i):
            rv_series.append(rv_i)

    if len(rv_series) < B_DIST_WINDOW // 2:
        return None

    p33 = float(np.percentile(rv_series, 33))
    p67 = float(np.percentile(rv_series, 67))

    if current_rv <= p33:
        # Calm: high SPY
        return {"SPY": B_EXPOSURE_CALM}
    elif current_rv >= p67:
        # Stressed: lower SPY + TLT
        return {"SPY": B_EXPOSURE_STRESSED, "TLT": B_TLT_STRESSED}
    else:
        # Middle: medium SPY
        return {"SPY": B_EXPOSURE_MID}


def _component_c_weights(ctx: BarContext) -> dict[str, float] | None:
    """Halloween seasonal: SPY Nov-Apr, TLT May-Oct."""
    if ctx.idx < 5:
        return None
    month = ctx.timestamp.month
    if month in C_WINTER_MONTHS:
        return {"SPY": C_EXPOSURE}
    else:
        return {"TLT": C_EXPOSURE}


class EnsembleCreditVolcarrySeasonal(Strategy):
    """Equal-weight ensemble of JNK credit gate + vol carry + Halloween seasonal."""

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
        for comp_weights in (a_w, b_w, c_w):
            if comp_weights is None:
                continue
            for sym, w in comp_weights.items():
                target[sym] = target.get(sym, 0.0) + w * self.component_weight

        if not target:
            return []

        # Cap total exposure
        total_w = sum(target.values())
        if total_w > self.exposure_cap and total_w > 0:
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

        # Sells first
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -MIN_DELTA:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))

        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > MIN_DELTA:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))

        return orders


UNIVERSE = ["QQQ", "SPY", "TLT", "GLD", "JNK"]

NAME = "ensemble_credit_volcarry_seasonal"
HYPOTHESIS = (
    "Equal-weight ensemble of 3 orthogonal diversifiers: "
    "A=JNK 30d SMA credit gate (QQQ risk-on / TLT risk-off), "
    "B=SPY realized-vol carry 21d RV vs 90d distribution (high SPY calm / SPY+TLT stressed), "
    "C=Halloween seasonal (SPY Nov-Apr / TLT May-Oct); "
    "each 1/3 capital, overlapping positions netted, exposure capped 0.97; "
    "all-ETF components (no individual stock selection)"
)

STRATEGY = EnsembleCreditVolcarrySeasonal()
