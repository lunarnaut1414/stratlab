"""Long-End Slope + Credit Dual Regime with SP500 Momentum — gen_9 sonnet-10

Hypothesis: Combine TYX-TNX long-end yield slope (vs 100d MA) AND JNK 30d SMA
credit gate into a 4-state allocator where the best regime holds SP500 momentum:

4 states:
  1. Steep slope (TYX-TNX > 100d MA) + Credit OK (JNK > 30d MA):
     → top-15 SP500 by 63d momentum, equal-weight (97% exposure)
     → "Full risk-on": growth regime confirmed by both signals
  2. Steep slope + Credit Stressed (JNK < 30d MA):
     → QQQ 60% + IEF 37%
     → Tech defensive: slope still growth-positive but credit warns
  3. Flat/Inverted slope + Credit OK:
     → SPY 60% + TLT 37%
     → Equity but with duration protection
  4. Flat/Inverted slope + Credit Stressed:
     → TLT 97%
     → Full defensive: both signals negative

SPY 200d SMA outer bear gate: if SPY below 200d SMA, force TLT regardless.
Biweekly rebalance (10 bars) — matches opus1_longend_slope timing.

Rationale: opus1_longend_slope_equity_gate (gen_8 best OOS at 0.79) uses slope
alone to gate SP500 momentum. This variant adds JNK credit as a second dimension,
creating differentiated neutral-regime behavior. When slope is steep but credit
is stressed (rare combo — rates pricing growth but corporate spreads worried), the
QQQ defensive allocation captures tech's relative safety. The 4-state grid creates
timing that differs from all single-signal strategies.

Differentiation:
- gen8_opus1_longend_slope_equity_gate: slope ONLY, no credit gate, all 4 states
  go to SP500 stocks or SPY+TLT blend
- gen8_sp500_credit_zscore_3tier: credit zscore ONLY, no slope gate
- gen6_jnk_vix_dual_gate_qqq: credit + VIX (not slope), different defensive routing
- This adds slope as a SECOND DIMENSION to credit, not previously done
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# Parameters
SLOPE_MA_WINDOW = 100    # MA for TYX-TNX long-end slope
JNK_MA_WINDOW = 30       # MA for JNK credit gate
MOMENTUM_WINDOW = 63     # 63d SP500 momentum
TREND_WINDOW = 200       # SPY 200d SMA outer bear gate
TOP_K = 15
REBALANCE_EVERY = 10     # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["JNK", "QQQ", "SPY", "IEF", "TLT", "^TYX", "^TNX"]


UNIVERSE = _universe


class SlopeCreditSP500Hybrid(Strategy):
    """4-state allocator: long-end slope x JNK credit, top state = SP500 momentum."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, SLOPE_MA_WINDOW, MOMENTUM_WINDOW) + 10
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
            # --- Signal A: Long-end slope (TYX - TNX) vs 100d MA ---
            tyx_hist = ctx.history("^TYX")
            tnx_hist = ctx.history("^TNX")

            slope_positive = True  # default: steep (growth regime)
            if len(tyx_hist) >= SLOPE_MA_WINDOW + 5 and len(tnx_hist) >= SLOPE_MA_WINDOW + 5:
                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()

                if len(tyx_close) >= SLOPE_MA_WINDOW and len(tnx_close) >= SLOPE_MA_WINDOW:
                    min_len = min(len(tyx_close), len(tnx_close))
                    tyx_vals = tyx_close.values[-min_len:]
                    tnx_vals = tnx_close.values[-min_len:]
                    slope_series = tyx_vals - tnx_vals
                    slope_now = float(slope_series[-1])
                    slope_ma = float(np.mean(slope_series[-SLOPE_MA_WINDOW:]))
                    slope_positive = slope_now > slope_ma

            # --- Signal B: JNK vs 30d MA ---
            jnk_hist = ctx.history("JNK")
            credit_ok = True  # default: credit benign
            if len(jnk_hist) >= JNK_MA_WINDOW + 5:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= JNK_MA_WINDOW:
                    jnk_ma = float(jnk_close.iloc[-JNK_MA_WINDOW:].mean())
                    jnk_price = live_all.get("JNK", 0.0)
                    if jnk_price > 0:
                        credit_ok = jnk_price >= jnk_ma

            # --- 4-state allocation ---
            if slope_positive and credit_ok:
                # Full risk-on: SP500 momentum
                target = self._select_sp500(ctx, live_all)
            elif slope_positive and not credit_ok:
                # Slope positive but credit stressed: tech defensive
                target = {"QQQ": 0.60, "IEF": 0.37}
            elif not slope_positive and credit_ok:
                # Flat slope but credit ok: equity + duration blend
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

    def _select_sp500(
        self,
        ctx: BarContext,
        live_all: dict[str, float],
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by 63d momentum above 200d SMA."""
        prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
        if len(prices_window) < MOMENTUM_WINDOW:
            return {"IEF": EXPOSURE}

        exclude = {"JNK", "QQQ", "SPY", "IEF", "TLT"}
        live = {s: p for s, p in live_all.items()
                if s not in exclude and not s.startswith("^")}

        scores: dict[str, float] = {}
        for sym in live:
            if sym not in prices_window.columns:
                continue
            col = prices_window[sym].dropna()
            if len(col) < MOMENTUM_WINDOW:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-MOMENTUM_WINDOW])
            if p_start <= 0:
                continue
            r = p_end / p_start - 1.0
            if np.isfinite(r):
                scores[sym] = r

        if len(scores) < TOP_K:
            return {"IEF": EXPOSURE}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for sym, _ in ranked:
            if len(selected) >= TOP_K:
                break
            hist = ctx.history(sym)
            if len(hist) < TREND_WINDOW:
                continue
            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
            price = live.get(sym, 0.0)
            if price > sma:
                selected.append(sym)

        if not selected:
            return {"IEF": EXPOSURE}

        per_weight = EXPOSURE / len(selected)
        return {sym: per_weight for sym in selected}


NAME = "slope_credit_sp500_hybrid"
HYPOTHESIS = (
    "Long-end yield slope (TYX-TNX vs 100d MA) + JNK credit 4-state allocator with SP500 "
    "momentum in top state: steep+credit-ok → top-15 SP500 63d momentum; steep+credit-stressed "
    "→ QQQ 60%+IEF 37%; flat+credit-ok → SPY 60%+TLT 37%; flat+credit-stressed → TLT 97%; "
    "SPY 200d bear override to TLT; biweekly rebalance"
)

STRATEGY = SlopeCreditSP500Hybrid()
