"""Long-End Yield Slope + Credit Dual Regime Allocator — gen_9 sonnet-10

Hypothesis: Combine two independent macro signals into a 4-state allocation:
- Signal A: TYX-TNX long-end yield slope (30Y minus 10Y yield) vs 100d MA
  - Steep slope (TYX-TNX > 100d MA): growth/inflation regime → equity-positive
  - Flat/inverted slope: flight-to-quality → equity-negative
- Signal B: JNK vs its 30d SMA (credit spread regime)
  - JNK > 30d MA: credit benign → risk assets favored
  - JNK < 30d MA: credit stress → defensive

4 states:
  1. Steep + Credit OK → QQQ 97% (growth/tech leadership)
  2. Steep + Credit Stressed → SPY 60% + IEF 37% (broad equity, partial defense)
  3. Flat/Inv + Credit OK → SPY 60% + TLT 37% (blend, duration helps)
  4. Flat/Inv + Credit Stressed → TLT 97% (full defensive)

SPY 200d SMA outer bear gate: if SPY below 200d SMA, force TLT regardless.
Weekly rebalance (every 5 bars) for adequate trade count.

Rationale: Long-end slope captures duration risk premium dynamics; credit spread
captures credit cycle. When both are positive (steep slope + credit benign), the
growth regime is strongest → QQQ. Each signal alone drives a different prior art:
- gen8_opus1_longend_slope_equity_gate: slope-only gating SP500 stocks
- gen6_jnk_vix_dual_gate_qqq: credit+VIX gating QQQ/SPY/TLT
This combines slope with credit (not VIX) in a 4-state grid — not done before.

Differentiation:
- Distinct from all SP500 cross-sectional momentum strategies (no stock selection)
- Distinct from existing slope strategies (adds credit gate creating 4 states)
- Distinct from existing credit strategies (adds slope gate as dimension)
- Pure ETF exposure (QQQ/SPY/IEF/TLT) → guaranteed low stock-selection corr
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Parameters
SLOPE_MA_WINDOW = 100    # MA for TYX-TNX long-end slope
JNK_MA_WINDOW = 30       # MA for JNK credit gate
TREND_WINDOW = 200       # SPY 200d SMA outer bear gate
REBALANCE_EVERY = 5      # Weekly rebalance
EXPOSURE = 0.97

UNIVERSE = ["QQQ", "SPY", "IEF", "TLT", "JNK", "^TYX", "^TNX"]


class SlopeCreditDualRegime(Strategy):
    """4-state allocator: long-end slope x JNK credit dual-signal regime."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, SLOPE_MA_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA outer bear gate ---
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < TREND_WINDOW:
            return []
        spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
        spy_price = live_all.get("SPY", 0.0)
        spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target: dict[str, float] = {"TLT": EXPOSURE}
        else:
            # --- Signal A: Long-end yield slope (TYX - TNX) vs 100d MA ---
            tyx_hist = ctx.history("^TYX")  # 30-year yield
            tnx_hist = ctx.history("^TNX")  # 10-year yield

            slope_positive = True  # default: assume steep (growth regime)
            if len(tyx_hist) >= SLOPE_MA_WINDOW + 5 and len(tnx_hist) >= SLOPE_MA_WINDOW + 5:
                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()

                if len(tyx_close) >= SLOPE_MA_WINDOW and len(tnx_close) >= SLOPE_MA_WINDOW:
                    # Align to same length
                    min_len = min(len(tyx_close), len(tnx_close))
                    tyx_aligned = tyx_close.values[-min_len:]
                    tnx_aligned = tnx_close.values[-min_len:]

                    # Current slope value
                    slope_now = float(tyx_aligned[-1] - tnx_aligned[-1])

                    # MA of slope over SLOPE_MA_WINDOW
                    slope_series = tyx_aligned - tnx_aligned
                    if len(slope_series) >= SLOPE_MA_WINDOW:
                        slope_ma = float(np.mean(slope_series[-SLOPE_MA_WINDOW:]))
                        slope_positive = slope_now > slope_ma
                    else:
                        slope_positive = slope_now > 0  # fallback: compare to zero

            # --- Signal B: JNK vs 30d MA ---
            jnk_hist = ctx.history("JNK")
            credit_ok = True  # default: assume credit benign
            if len(jnk_hist) >= JNK_MA_WINDOW + 5:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= JNK_MA_WINDOW:
                    jnk_ma = float(jnk_close.iloc[-JNK_MA_WINDOW:].mean())
                    jnk_price = live_all.get("JNK", 0.0)
                    if jnk_price > 0:
                        credit_ok = jnk_price >= jnk_ma

            # --- 4-state allocation ---
            if slope_positive and credit_ok:
                # Both positive: maximum risk-on → QQQ
                target = {"QQQ": EXPOSURE}
            elif slope_positive and not credit_ok:
                # Yield environment positive but credit stressed: reduce risk
                target = {"SPY": 0.60, "IEF": 0.37}
            elif not slope_positive and credit_ok:
                # Yield environment flat/inverted but credit still ok:
                # blend equities with duration
                target = {"SPY": 0.60, "TLT": 0.37}
            else:
                # Both negative: full defensive
                target = {"TLT": EXPOSURE}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "slope_credit_dual_regime"
HYPOTHESIS = (
    "Long-end yield slope (TYX-TNX vs 100d MA) PLUS JNK 30d SMA credit gate: "
    "steep+credit-ok → QQQ 97%; steep+credit-stressed → SPY 60%+IEF 37%; "
    "flat+credit-ok → SPY 60%+TLT 37%; flat+credit-stressed → TLT 97%; "
    "SPY 200d bear override to TLT; weekly rebalance; pure ETF no stock selection"
)

STRATEGY = SlopeCreditDualRegime()
