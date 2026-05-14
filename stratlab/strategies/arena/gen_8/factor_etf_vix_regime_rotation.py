"""Factor ETF VIX-Regime Rotation — gen_8 sonnet-9

Hypothesis: Rotate among factor ETFs based on VIX regime:
- VIX calm (VIX < 20): hold whichever of MTUM or QUAL has stronger 42d
  momentum, sized to 97%.
- VIX elevated (20 <= VIX < 28): hold SPLV (low-volatility factor) at 97%
  as defensive-equity exposure.
- VIX stress (VIX >= 28) OR SPY below 200d SMA: hold TLT at 97%.

Rebalance every 5 bars (weekly). Signal updates are VIX-level driven not
calendar-driven.

Rationale: The factor cycle has a well-documented relationship to the
volatility regime. Momentum (MTUM) and quality (QUAL) outperform in
low-vol expansion. Low-volatility factor (SPLV) is designed precisely for
elevated-vol environments — it holds the least-volatile S&P500 stocks which
drawdown less in turbulence. This creates a smooth factor hand-off that
avoids the binary risk-on/risk-off rotation problem.

Differentiation: The gen5 Factor ETF rotation (MTUM/QUAL/IVE/USMV by 3m
return, IS Calmar 0.38) failed due to no regime gate — all factor ETFs
suffer in drawdowns equally. This version adds VIX-tiered regime gating to
solve that. SPLV as the intermediate defensive layer is novel on the
leaderboard (no strategy uses SPLV as primary holding). The two-factor
switcher MTUM vs QUAL in calm regime adds momentum differentiation above
buy-and-hold MTUM.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 42      # 42-day momentum for MTUM vs QUAL comparison
TREND_WINDOW = 200        # SPY 200d SMA bear gate
VIX_CALM = 20.0           # Below this: momentum/quality factor
VIX_ELEVATED = 28.0       # Above this: full defensive (TLT)
REBALANCE_DAYS = 5        # Weekly
EXPOSURE = 0.97

UNIVERSE = [
    "^VIX",   # Regime signal (non-tradeable)
    "SPY",    # Bear gate reference
    "MTUM",   # Momentum factor ETF
    "QUAL",   # Quality factor ETF
    "SPLV",   # Low-volatility factor ETF
    "TLT",    # Long bonds (stress defensive)
]


class FactorEtfVixRegimeRotation(Strategy):
    """Factor ETF rotation: MTUM/QUAL in calm, SPLV in elevated VIX, TLT in stress."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, MOMENTUM_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- VIX level for regime ---
        vix_hist = ctx.history("^VIX")
        vix_val = 20.0  # default neutral
        if len(vix_hist) >= 2:
            vix_val = float(vix_hist["close"].iloc[-1])

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live.get("SPY", 0.0)
            spy_bear = (spy_price > 0) and (spy_price <= spy_sma)

        # --- Determine target based on VIX regime ---
        if spy_bear or vix_val >= VIX_ELEVATED:
            # Stress or bear market: full TLT
            target = {"TLT": EXPOSURE}
        elif vix_val >= VIX_CALM:
            # Elevated VIX: low-volatility factor ETF
            if "SPLV" in live and live["SPLV"] > 0:
                target = {"SPLV": EXPOSURE}
            else:
                target = {"TLT": EXPOSURE}
        else:
            # Calm VIX: compare MTUM vs QUAL by 42-day momentum
            prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
            if len(prices_window) < MOMENTUM_WINDOW:
                return []

            def get_momentum(sym: str) -> float | None:
                if sym not in prices_window.columns:
                    return None
                col = prices_window[sym].dropna()
                if len(col) < MOMENTUM_WINDOW:
                    return None
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOMENTUM_WINDOW])
                if p_start <= 0:
                    return None
                ret = p_end / p_start - 1.0
                return ret if np.isfinite(ret) else None

            mtum_mom = get_momentum("MTUM")
            qual_mom = get_momentum("QUAL")

            if mtum_mom is None and qual_mom is None:
                target = {"TLT": EXPOSURE}
            elif qual_mom is None or (mtum_mom is not None and mtum_mom >= qual_mom):
                target = {"MTUM": EXPOSURE}
            else:
                target = {"QUAL": EXPOSURE}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym, 0.0)
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


NAME = "factor_etf_vix_regime_rotation"
HYPOTHESIS = (
    "Factor ETF VIX-regime rotation: VIX<20 hold stronger-momentum of MTUM/QUAL (42d); "
    "VIX 20-28 hold SPLV (low-vol factor); VIX>=28 or SPY<200d SMA hold TLT; "
    "weekly rebalance; VIX-tiered factor rotation with SPLV intermediate layer is novel."
)

STRATEGY = FactorEtfVixRegimeRotation()
