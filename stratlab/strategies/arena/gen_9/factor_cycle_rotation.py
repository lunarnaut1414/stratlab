"""Factor Cycle Rotation — gen_9 sonnet-10

Hypothesis: Use the VTV (Vanguard Value ETF) vs VUG (Vanguard Growth ETF) 42-day
return spread as a factor-cycle regime signal. When growth leads (VUG > VTV on 42d
return), the environment favors high-momentum growth stocks — hold top-15 SP500 by
63d momentum. When value leads (VTV > VUG on 42d return), the environment typically
features mean reversion and quality/value factor dominance — hold top-15 SP500 by
126d-skip-21d momentum (the classic Jegadeesh-Titman skip-month horizon, which
reduces reversal contamination and favors longer-duration winners).

SPY 200d SMA bear gate: in bear market, hold TLT regardless of factor cycle.
Biweekly rebalance (10 bars).

Rationale: Factor cycles (growth vs value) are well-documented in academic literature
and create distinct stock selection conditions. The 42d spread window is long enough
to be stable but short enough to respond to regime shifts like rate spikes (which
rotate from growth to value). Crucially, this signal is orthogonal to:
- JNK/LQD credit signals (already dominant in leaderboard)
- VIX regime signals
- Yield curve slope signals
- Dollar (UUP) signals

VTV launched 2004-01-30, VUG launched 2004-01-30 — both cover IS (2010-2018).

The skip-month variant for value regimes reduces reversal contamination and produces
different stock rankings than the standard 63d window, creating within-strategy
diversification across factor regimes.

Differentiation from leaderboard: no prior strategy uses VTV/VUG factor-cycle spread.
The skip-month momentum was used in gen8_sp500_skipmon_63sma_momentum but without
a factor-cycle routing layer.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# Parameters
FACTOR_WINDOW = 42        # Window for IWD vs IWF return spread
MOM_GROWTH = 63           # Momentum window when growth leads
MOM_VALUE_LONG = 126      # Skip-month lookback when value leads
MOM_VALUE_SKIP = 21       # Skip days for Jegadeesh-Titman skip-month
TREND_WINDOW = 200        # SPY 200d SMA bear gate
TOP_K = 15
REBALANCE_EVERY = 10      # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["VTV", "VUG", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class FactorCycleRotation(Strategy):
    """SP500 momentum routing via IWD/IWF factor-cycle regime signal."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, MOM_VALUE_LONG + MOM_VALUE_SKIP) + 10
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < TREND_WINDOW:
            return []
        spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
        spy_price = live_all.get("SPY", 0.0)
        spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target: dict[str, float] = {"TLT": EXPOSURE}
        else:
            # --- Compute VTV vs VUG factor spread ---
            vtv_hist = ctx.history("VTV")
            vug_hist = ctx.history("VUG")

            growth_regime = True  # default to growth if signal unavailable
            if len(vtv_hist) >= FACTOR_WINDOW + 5 and len(vug_hist) >= FACTOR_WINDOW + 5:
                vtv_close = vtv_hist["close"].dropna()
                vug_close = vug_hist["close"].dropna()

                if len(vtv_close) >= FACTOR_WINDOW and len(vug_close) >= FACTOR_WINDOW:
                    vtv_ret = float(vtv_close.iloc[-1] / vtv_close.iloc[-FACTOR_WINDOW] - 1.0)
                    vug_ret = float(vug_close.iloc[-1] / vug_close.iloc[-FACTOR_WINDOW] - 1.0)

                    if np.isfinite(vtv_ret) and np.isfinite(vug_ret):
                        # Growth regime: VUG (growth) leads VTV (value)
                        growth_regime = vug_ret >= vtv_ret

            # --- Select momentum window based on factor regime ---
            if growth_regime:
                mom_window = MOM_GROWTH
                prices_window = ctx.closes_window(mom_window + 5)
                if len(prices_window) < mom_window:
                    target = {"IEF": EXPOSURE}
                else:
                    target = self._select_sp500(ctx, closes, live_all, prices_window, mom_window)
            else:
                # Value regime: use skip-month Jegadeesh-Titman momentum
                need = MOM_VALUE_LONG + MOM_VALUE_SKIP + 5
                prices_window = ctx.closes_window(need)
                if len(prices_window) < MOM_VALUE_LONG:
                    target = {"IEF": EXPOSURE}
                else:
                    target = self._select_sp500_skipmon(ctx, closes, live_all, prices_window)

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
        closes: object,
        live_all: dict[str, float],
        prices_window: object,
        mom_window: int,
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by simple momentum."""
        exclude = {"VTV", "VUG", "TLT", "IEF", "SPY"}
        live = {s: p for s, p in live_all.items() if s not in exclude}

        scores: dict[str, float] = {}
        for sym in live:
            if sym not in prices_window.columns:
                continue
            col = prices_window[sym].dropna()
            if len(col) < mom_window:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-mom_window])
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

    def _select_sp500_skipmon(
        self,
        ctx: BarContext,
        closes: object,
        live_all: dict[str, float],
        prices_window: object,
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by skip-month momentum (126d skip 21d)."""
        exclude = {"VTV", "VUG", "TLT", "IEF", "SPY"}
        live = {s: p for s, p in live_all.items() if s not in exclude}

        scores: dict[str, float] = {}
        for sym in live:
            if sym not in prices_window.columns:
                continue
            col = prices_window[sym].dropna()
            # Need at least MOM_VALUE_LONG + 1 bars
            if len(col) < MOM_VALUE_LONG + MOM_VALUE_SKIP:
                continue
            # Return from [-MOM_VALUE_LONG - MOM_VALUE_SKIP] to [-MOM_VALUE_SKIP]
            p_end = float(col.iloc[-MOM_VALUE_SKIP])
            p_start = float(col.iloc[-MOM_VALUE_LONG - MOM_VALUE_SKIP])
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


NAME = "factor_cycle_rotation"
HYPOTHESIS = (
    "SP500 factor-cycle rotation: use VTV (Vanguard Value ETF) vs VUG (Vanguard Growth ETF) "
    "42d return spread as factor regime; when growth leads hold top-15 SP500 stocks by 63d "
    "momentum; when value leads hold top-15 SP500 stocks by 126d-skip-21d momentum "
    "(value-friendly horizon); SPY 200d bear gate to TLT; biweekly rebalance"
)

STRATEGY = FactorCycleRotation()
