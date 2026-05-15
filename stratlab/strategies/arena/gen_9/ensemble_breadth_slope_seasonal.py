"""Ensemble: Breadth + Slope/Credit + Seasonal — gen_9 sonnet-10

Three-component ensemble combining orthogonal regime signals:

Component A — Breadth-gated SP500 momentum (1/3 capital):
  Uses market breadth (fraction of SP500 stocks above 50d SMA) percentile rank.
  When breadth > 60th pct: top-15 SP500 by 63d momentum.
  When breadth < 40th pct: IEF 60%+TLT 37%.
  Middle: top-15 at 50% + IEF 47%.
  SPY 200d bear gate to TLT.

Component B — Long-end slope + credit dual regime (1/3 capital):
  Steep TYX-TNX slope (vs 100d MA) + JNK credit (vs 30d MA).
  Both positive: top-15 SP500 by 63d momentum.
  Slope positive, credit stressed: QQQ 60%+IEF 37%.
  Slope flat/inverted, credit ok: SPY 60%+TLT 37%.
  Both negative: TLT 97%.
  SPY 200d bear gate to TLT.

Component C — Sell-in-May seasonal (1/3 capital):
  SPY Nov-Apr (winter); TLT May-Oct (summer).
  Classic seasonal anomaly, purely calendar-driven, structurally orthogonal.

Why this triplet:
  A uses WITHIN-MARKET breadth signal (distinct from macro signals)
  B uses LONG-END YIELD CURVE + CREDIT (macro/rate signals)
  C uses CALENDAR (no market data, purely exogenous)
  Three orthogonal signal types → low pairwise correlation → diversification benefit.

Rationale: gen_5 ensemble_bond_credit_seasonal (IS Calmar 0.68, OOS Calmar 0.53)
shows that combining 3 low-corr signals at 1/3 each reliably produces higher
Calmar than individual components. This ensemble uses better individual components:
- Breadth (IS Calmar 0.61, corr 0.633) vs gen_5's bond_equity_regime
- Slope+credit (IS Calmar 0.62, corr 0.601) vs gen_5's JNK/LQD MA crossover
- Same seasonal component as gen_5 (tested, calendar-driven)

The combination of independently-designed IS Calmar ~0.6 strategies with pairwise
low correlation should target IS Calmar 0.7+ with reduced drawdown.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# Component A — Breadth parameters
A_STOCK_SMA = 50
A_BREADTH_HIST = 252
A_BREADTH_BULL = 0.60
A_BREADTH_BEAR = 0.40
A_MOM_WIN = 63
A_TREND_WIN = 200
A_TOP_K = 15
A_EXPOSURE_BULL = 0.97
A_EXPOSURE_NEUTRAL = 0.50
A_REBALANCE = 10

# Component B — Slope+Credit parameters
B_SLOPE_MA = 100
B_JNK_MA = 30
B_MOM_WIN = 63
B_TREND_WIN = 200
B_TOP_K = 15
B_EXPOSURE = 0.97
B_REBALANCE = 10

# Component C — Seasonal parameters
C_WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

# Ensemble
COMPONENT_WEIGHT = 1.0 / 3.0
EXPOSURE_CAP = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["JNK", "QQQ", "SPY", "IEF", "TLT", "^TYX", "^TNX"]


UNIVERSE = _universe


class EnsembleBreadthSlopeSeasonal(Strategy):
    """3-component ensemble: breadth + slope/credit + seasonal."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)
        self._breadth_history: list[float] = []
        self._rebalance_a = A_REBALANCE
        self._rebalance_b = B_REBALANCE

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Update breadth history every bar
        self._update_breadth(ctx)

        warmup = max(A_TREND_WIN, B_TREND_WIN, A_BREADTH_HIST, B_SLOPE_MA) + 10
        if ctx.idx < warmup:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # Only rebalance if either component A or B needs rebalancing
        # (C is checked per bar via calendar)
        a_rebalance = (ctx.idx % self._rebalance_a == 0)
        b_rebalance = (ctx.idx % self._rebalance_b == 0)
        c_rebalance = True  # seasonal updates every bar that's a month boundary

        if not (a_rebalance or b_rebalance):
            # Check if we're at a month boundary for seasonal
            if ctx.idx > 0:
                prev_idx = ctx.idx - 1
                # Only rebalance at month changes (approximate with every 5 bars check)
                if ctx.idx % 5 != 0:
                    return []

        # Compute component weights
        weights_a = self._component_a(ctx, live_all) if a_rebalance else {}
        weights_b = self._component_b(ctx, live_all) if b_rebalance else {}
        weights_c = self._component_c(ctx, live_all)

        # Net weights across components
        # Only update if component was rebalanced
        # For simplicity, recalculate all components on any rebalance bar
        if not (a_rebalance or b_rebalance or ctx.idx % 5 == 0):
            return []

        # Recalculate all components
        weights_a = self._component_a(ctx, live_all)
        weights_b = self._component_b(ctx, live_all)
        weights_c = self._component_c(ctx, live_all)

        # Combine at 1/3 each
        combined: dict[str, float] = {}
        for sym, w in weights_a.items():
            combined[sym] = combined.get(sym, 0.0) + w * COMPONENT_WEIGHT
        for sym, w in weights_b.items():
            combined[sym] = combined.get(sym, 0.0) + w * COMPONENT_WEIGHT
        for sym, w in weights_c.items():
            combined[sym] = combined.get(sym, 0.0) + w * COMPONENT_WEIGHT

        # Cap total exposure
        total_weight = sum(combined.values())
        if total_weight > EXPOSURE_CAP:
            scale = EXPOSURE_CAP / total_weight
            combined = {s: w * scale for s, w in combined.items()}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in combined and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in combined.items():
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

    # ---- Component A: Breadth-gated SP500 momentum ----

    def _update_breadth(self, ctx: BarContext) -> None:
        """Update rolling breadth (fraction of SP500 stocks above 50d SMA)."""
        prices_window = ctx.closes_window(A_STOCK_SMA + 5)
        if len(prices_window) < A_STOCK_SMA:
            return
        exclude = {"JNK", "QQQ", "SPY", "IEF", "TLT"}
        above = 0
        total = 0
        for sym in prices_window.columns:
            if sym in exclude or sym.startswith("^"):
                continue
            col = prices_window[sym].dropna()
            if len(col) < A_STOCK_SMA:
                continue
            sma = float(col.iloc[-A_STOCK_SMA:].mean())
            price = float(col.iloc[-1])
            total += 1
            if price > sma:
                above += 1
        if total >= 50:
            self._breadth_history.append(above / total)
            if len(self._breadth_history) > A_BREADTH_HIST + 10:
                self._breadth_history = self._breadth_history[-(A_BREADTH_HIST + 10):]

    def _component_a(self, ctx: BarContext, live_all: dict[str, float]) -> dict[str, float]:
        """Breadth-gated SP500 momentum component."""
        # SPY bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < A_TREND_WIN:
            return {"TLT": 1.0}
        spy_sma = float(spy_hist["close"].iloc[-A_TREND_WIN:].mean())
        spy_price = live_all.get("SPY", 0.0)
        if spy_price > 0 and spy_price <= spy_sma:
            return {"TLT": 1.0}

        # Breadth percentile rank
        if len(self._breadth_history) < 30:
            regime = "neutral"
        else:
            hist = self._breadth_history[-A_BREADTH_HIST:]
            current = hist[-1]
            below = sum(1 for b in hist[:-1] if b <= current)
            pct = below / max(len(hist) - 1, 1)
            if pct >= A_BREADTH_BULL:
                regime = "bull"
            elif pct <= A_BREADTH_BEAR:
                regime = "bear"
            else:
                regime = "neutral"

        if regime == "bear":
            return {"IEF": 0.60, "TLT": 0.37}
        elif regime == "bull":
            return self._select_sp500(ctx, live_all, A_TOP_K, A_EXPOSURE_BULL)
        else:
            stocks = self._select_sp500(ctx, live_all, A_TOP_K, A_EXPOSURE_NEUTRAL)
            if "IEF" in stocks:
                return {"IEF": 1.0}
            stocks["IEF"] = 1.0 - A_EXPOSURE_NEUTRAL
            return stocks

    # ---- Component B: Long-end slope + credit regime ----

    def _component_b(self, ctx: BarContext, live_all: dict[str, float]) -> dict[str, float]:
        """Slope+credit 4-state allocator component."""
        # SPY bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < B_TREND_WIN:
            return {"TLT": 1.0}
        spy_sma = float(spy_hist["close"].iloc[-B_TREND_WIN:].mean())
        spy_price = live_all.get("SPY", 0.0)
        if spy_price > 0 and spy_price <= spy_sma:
            return {"TLT": 1.0}

        # Long-end slope signal
        slope_positive = True
        tyx_hist = ctx.history("^TYX")
        tnx_hist = ctx.history("^TNX")
        if len(tyx_hist) >= B_SLOPE_MA + 5 and len(tnx_hist) >= B_SLOPE_MA + 5:
            tyx_close = tyx_hist["close"].dropna()
            tnx_close = tnx_hist["close"].dropna()
            if len(tyx_close) >= B_SLOPE_MA and len(tnx_close) >= B_SLOPE_MA:
                min_len = min(len(tyx_close), len(tnx_close))
                slope = tyx_close.values[-min_len:] - tnx_close.values[-min_len:]
                slope_positive = float(slope[-1]) > float(np.mean(slope[-B_SLOPE_MA:]))

        # JNK credit signal
        credit_ok = True
        jnk_hist = ctx.history("JNK")
        if len(jnk_hist) >= B_JNK_MA + 5:
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= B_JNK_MA:
                jnk_ma = float(jnk_close.iloc[-B_JNK_MA:].mean())
                jnk_price = live_all.get("JNK", 0.0)
                if jnk_price > 0:
                    credit_ok = jnk_price >= jnk_ma

        if slope_positive and credit_ok:
            return self._select_sp500(ctx, live_all, B_TOP_K, B_EXPOSURE)
        elif slope_positive and not credit_ok:
            return {"QQQ": 0.60, "IEF": 0.40}
        elif not slope_positive and credit_ok:
            return {"SPY": 0.60, "TLT": 0.40}
        else:
            return {"TLT": 1.0}

    # ---- Component C: Seasonal ----

    def _component_c(self, ctx: BarContext, live_all: dict[str, float]) -> dict[str, float]:
        """Sell-in-May calendar seasonal component."""
        month = ctx.timestamp.month
        if month in C_WINTER_MONTHS:
            return {"SPY": 1.0}
        else:
            return {"TLT": 1.0}

    # ---- Shared utility ----

    def _select_sp500(
        self,
        ctx: BarContext,
        live_all: dict[str, float],
        top_k: int,
        exposure: float,
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by 63d momentum above 200d SMA."""
        prices_window = ctx.closes_window(A_MOM_WIN + 5)
        if len(prices_window) < A_MOM_WIN:
            return {"IEF": 1.0}
        exclude = {"JNK", "QQQ", "SPY", "IEF", "TLT"}
        live = {s: p for s, p in live_all.items()
                if s not in exclude and not s.startswith("^")}
        scores: dict[str, float] = {}
        for sym in live:
            if sym not in prices_window.columns:
                continue
            col = prices_window[sym].dropna()
            if len(col) < A_MOM_WIN:
                continue
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-A_MOM_WIN])
            if p_start <= 0:
                continue
            r = p_end / p_start - 1.0
            if np.isfinite(r):
                scores[sym] = r
        if len(scores) < top_k:
            return {"IEF": 1.0}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for sym, _ in ranked:
            if len(selected) >= top_k:
                break
            hist = ctx.history(sym)
            if len(hist) < A_TREND_WIN:
                continue
            sma = float(hist["close"].iloc[-A_TREND_WIN:].mean())
            price = live.get(sym, 0.0)
            if price > sma:
                selected.append(sym)
        if not selected:
            return {"IEF": 1.0}
        per_weight = exposure / len(selected)
        return {sym: per_weight for sym in selected}


NAME = "ensemble_breadth_slope_seasonal"
HYPOTHESIS = (
    "Equal-weight ensemble of 3 orthogonal components: A=breadth-gated SP500 momentum "
    "(within-market breadth percentile rank), B=long-end slope+credit 4-state SP500 "
    "(TYX-TNX vs 100d MA + JNK 30d MA), C=sell-in-May seasonal (SPY winter/TLT summer); "
    "each 1/3 capital, exposure capped 0.97; orthogonal signals: breadth, macro rates, calendar"
)

STRATEGY = EnsembleBreadthSlopeSeasonal()
