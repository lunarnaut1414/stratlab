"""VIX Percentile-Rank Gated SP500 Momentum — gen_9 sonnet-10

Hypothesis: Use VIX's PERCENTILE RANK within a 252-day trailing window (not
absolute VIX level) as the regime gate for SP500 momentum. This adapts to
structural VIX regime shifts: what counts as "high VIX" changes over time.

Three regimes based on VIX percentile rank (0=lowest VIX in 252d, 1=highest):
  1. Bottom tercile (VIX pct < 0.33): Calm → top-15 SP500 by 63d momentum, 97%
  2. Middle tercile (0.33 ≤ VIX pct < 0.67): Neutral → top-10 SP500, 70% + IEF 27%
  3. Top tercile (VIX pct ≥ 0.67): Stressed → IEF 60% + TLT 37%

SPY 200d SMA outer bear gate: if SPY below 200d SMA, force TLT regardless.
Biweekly rebalance (10 bars).

Rationale: Using VIX percentile rank (not absolute level) avoids the failure mode
documented in gen_8 dead_ends: "Threshold form (absolute, not percentile-rank) was
fragile." The percentile-rank approach self-calibrates to the recent VIX environment —
e.g., in 2017-2018 when VIX was structurally low, absolute-level gates got stuck in
"calm" forever. With percentile ranking, the distribution naturally creates roughly
33% / 33% / 33% allocation across regimes regardless of the structural VIX level.

The NEUTRAL tier partial exposure (70% stocks, 10 names) provides graceful degradation
vs binary risk-on/risk-off approaches.

Differentiation from leaderboard:
- gen5_sp500_momentum_vix_sized: VIX LINEAR SCALING of exposure (not 3 tiers)
- gen5_vix_gated_sp500_momentum: absolute VIX threshold (25), not percentile rank
- No leaderboard strategy uses percentile-rank normalization for VIX
- The neutral-tier partial exposure (70%) is different from binary full-risk/full-cash
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# Parameters
VIX_HIST_WINDOW = 252      # Trailing window for VIX percentile rank
VIX_CALM_PCT = 0.33        # Below this percentile = calm (full equity)
VIX_STRESS_PCT = 0.67      # Above this percentile = stressed (defensive)
MOMENTUM_WINDOW = 63       # 63d SP500 momentum
TREND_WINDOW = 200         # SPY 200d SMA outer bear gate
TOP_K_CALM = 15            # Full equity: top-15 stocks
TOP_K_NEUTRAL = 10         # Neutral: top-10 stocks
EXPOSURE_CALM = 0.97       # Full exposure in calm regime
EXPOSURE_NEUTRAL_STOCK = 0.70  # Stock portion in neutral regime
EXPOSURE_NEUTRAL_IEF = 0.27    # IEF portion in neutral regime
REBALANCE_EVERY = 10       # Biweekly


def _universe() -> list[str]:
    return sp500_tickers() + ["^VIX", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class VixPctileSP500Momentum(Strategy):
    """SP500 63d momentum gated by VIX percentile rank with 3-tier regime."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)
        self._vix_history: list[float] = []

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Update VIX history every bar for accurate percentile ranking
        self._update_vix_history(ctx)

        warmup = max(TREND_WINDOW, MOMENTUM_WINDOW, VIX_HIST_WINDOW) + 10
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
            target: dict[str, float] = {"TLT": EXPOSURE_CALM}
        else:
            # --- VIX percentile rank regime ---
            vix_pct = self._get_vix_percentile()

            if vix_pct is None:
                # Not enough VIX history — use neutral regime
                regime = "neutral"
            elif vix_pct < VIX_CALM_PCT:
                regime = "calm"
            elif vix_pct >= VIX_STRESS_PCT:
                regime = "stressed"
            else:
                regime = "neutral"

            if regime == "stressed":
                target = {"IEF": 0.60, "TLT": 0.37}
            elif regime == "calm":
                target = self._select_stocks(ctx, live_all, TOP_K_CALM, EXPOSURE_CALM)
            else:
                # Neutral: partial equity + IEF
                stock_target = self._select_stocks(
                    ctx, live_all, TOP_K_NEUTRAL, EXPOSURE_NEUTRAL_STOCK
                )
                if "IEF" in stock_target:
                    # _select_stocks returned defensive
                    target = {"IEF": EXPOSURE_CALM}
                else:
                    target = stock_target
                    target["IEF"] = EXPOSURE_NEUTRAL_IEF

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

    def _update_vix_history(self, ctx: BarContext) -> None:
        """Store current VIX close in history buffer."""
        vix_hist = ctx.history("^VIX")
        if len(vix_hist) >= 1:
            vix_val = float(vix_hist["close"].iloc[-1])
            if np.isfinite(vix_val) and vix_val > 0:
                self._vix_history.append(vix_val)
                if len(self._vix_history) > VIX_HIST_WINDOW + 10:
                    self._vix_history = self._vix_history[-(VIX_HIST_WINDOW + 10):]

    def _get_vix_percentile(self) -> float | None:
        """Return VIX's percentile rank within trailing 252d distribution."""
        if len(self._vix_history) < 30:
            return None
        history = self._vix_history[-VIX_HIST_WINDOW:]
        current = history[-1]
        below_count = sum(1 for v in history[:-1] if v <= current)
        pct_rank = below_count / max(len(history) - 1, 1)
        return pct_rank

    def _select_stocks(
        self,
        ctx: BarContext,
        live_all: dict[str, float],
        top_k: int,
        exposure: float,
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by 63d momentum above 200d SMA."""
        prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
        if len(prices_window) < MOMENTUM_WINDOW:
            return {"IEF": exposure}

        exclude = {"TLT", "IEF", "SPY"}
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

        if len(scores) < top_k:
            return {"IEF": exposure}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = []
        for sym, _ in ranked:
            if len(selected) >= top_k:
                break
            hist = ctx.history(sym)
            if len(hist) < TREND_WINDOW:
                continue
            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
            price = live.get(sym, 0.0)
            if price > sma:
                selected.append(sym)

        if not selected:
            return {"IEF": exposure}

        per_weight = exposure / len(selected)
        return {sym: per_weight for sym in selected}


NAME = "vix_pctile_sp500_momentum"
HYPOTHESIS = (
    "VIX percentile-rank gated SP500 momentum: compute VIX's position within trailing "
    "252d distribution; VIX pct < 33% → top-15 SP500 by 63d momentum at 97%; "
    "33-67% → top-10 SP500 at 70%+IEF 27%; >67% → IEF 60%+TLT 37%; "
    "SPY 200d bear override to TLT; biweekly rebalance"
)

STRATEGY = VixPctileSP500Momentum()
