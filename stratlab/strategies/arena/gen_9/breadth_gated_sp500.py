"""SP500 Market Breadth Gated Momentum — gen_9 sonnet-10

Hypothesis: Compute the percentage of SP500 stocks trading above their own 50-day
SMA (market breadth ratio). Use a percentile-rank of this breadth ratio over a
252-day trailing window as the regime gate. When breadth is in the top 40th
percentile or above (broad market participation), hold top-15 SP500 stocks by
63d momentum. When breadth is in the bottom 40th percentile (deteriorating
breadth, narrow leadership), hold IEF 60%+TLT 37% as defensive. In between
(neutral zone), hold top-15 SP500 at 50% exposure + IEF at 47%.

SPY 200d SMA bear override: if SPY below 200d SMA, force TLT regardless.
Biweekly rebalance (10 bars).

Rationale: Market breadth is a fundamental technical indicator — when broad
participation validates the trend (most stocks above their 50d SMA), momentum
strategies work well. When leadership narrows (few stocks driving the index),
the strategy becomes fragile. The percentile-rank approach avoids the failure
mode of gen_8's RSP/SPY breadth (which used an absolute threshold that became
ineffective when mega-cap leadership intensified).

Differentiation from leaderboard:
- RSP/SPY breadth in gen_8 failed (22% OOS retention) because of absolute threshold
- This uses within-universe breadth computed directly on SP500 constituents
- Percentile-rank normalization adapts to structural regime shifts
- Completely different signal from credit (JNK), VIX, yield-curve, and dollar-trend
  signals dominating the existing leaderboard
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

# Parameters
STOCK_SMA_WINDOW = 50     # SMA window for individual stock breadth
BREADTH_HIST_WINDOW = 252  # Trailing window for breadth percentile rank
BREADTH_BULL_PCT = 0.60   # Above this percentile = broad participation
BREADTH_BEAR_PCT = 0.40   # Below this percentile = deteriorating breadth
MOMENTUM_WINDOW = 63      # 63d stock momentum
TREND_WINDOW = 200        # SPY 200d SMA gate
TOP_K = 15
REBALANCE_EVERY = 10      # Biweekly
EXPOSURE = 0.97
NEUTRAL_EQUITY = 0.50     # Reduced stock exposure in neutral breadth regime


def _universe() -> list[str]:
    return sp500_tickers() + ["TLT", "IEF", "SPY"]


UNIVERSE = _universe


class BreadthGatedSP500(Strategy):
    """SP500 63d momentum gated by market breadth percentile rank."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)
        # Ring buffer to store daily breadth values
        self._breadth_history: list[float] = []

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, BREADTH_HIST_WINDOW + STOCK_SMA_WINDOW) + 10
        if ctx.idx < warmup:
            # Still compute and store breadth even during warmup
            self._update_breadth(ctx)
            return []

        # Update breadth every bar (even on non-rebalance bars) for accuracy
        self._update_breadth(ctx)

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
            # --- Breadth regime determination ---
            breadth_pct_rank = self._get_breadth_percentile_rank()

            if breadth_pct_rank is None:
                # Not enough breadth history — use neutral regime
                regime = "neutral"
            elif breadth_pct_rank >= BREADTH_BULL_PCT:
                regime = "bull"
            elif breadth_pct_rank <= BREADTH_BEAR_PCT:
                regime = "bear"
            else:
                regime = "neutral"

            if regime == "bear":
                target = {"IEF": 0.60, "TLT": 0.37}
            elif regime == "bull":
                target = self._select_stocks(ctx, live_all, EXPOSURE)
            else:
                # Neutral: partial stocks + IEF
                stock_target = self._select_stocks(ctx, live_all, NEUTRAL_EQUITY)
                if "IEF" in stock_target:
                    # _select_stocks returned defensive — use full defensive
                    target = {"IEF": EXPOSURE}
                else:
                    target = stock_target
                    target["IEF"] = EXPOSURE - NEUTRAL_EQUITY

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

    def _update_breadth(self, ctx: BarContext) -> None:
        """Compute breadth ratio (fraction of SP500 stocks above 50d SMA) and store."""
        prices_window = ctx.closes_window(STOCK_SMA_WINDOW + 5)
        if len(prices_window) < STOCK_SMA_WINDOW:
            return

        above_count = 0
        total_count = 0
        for sym in prices_window.columns:
            # Skip ETFs and non-stock symbols
            if sym in ("TLT", "IEF", "SPY") or sym.startswith("^"):
                continue
            col = prices_window[sym].dropna()
            if len(col) < STOCK_SMA_WINDOW:
                continue
            sma_50 = float(col.iloc[-STOCK_SMA_WINDOW:].mean())
            current_price = float(col.iloc[-1])
            total_count += 1
            if current_price > sma_50:
                above_count += 1

        if total_count >= 50:  # Need minimum stocks to have a meaningful ratio
            breadth = above_count / total_count
            self._breadth_history.append(breadth)
            # Keep only what we need
            if len(self._breadth_history) > BREADTH_HIST_WINDOW + 10:
                self._breadth_history = self._breadth_history[-(BREADTH_HIST_WINDOW + 10):]

    def _get_breadth_percentile_rank(self) -> float | None:
        """Return percentile rank of current breadth within trailing 252 days."""
        if len(self._breadth_history) < 30:
            return None

        history = self._breadth_history[-BREADTH_HIST_WINDOW:]
        current = history[-1]
        below_count = sum(1 for b in history[:-1] if b <= current)
        pct_rank = below_count / max(len(history) - 1, 1)
        return pct_rank

    def _select_stocks(
        self,
        ctx: BarContext,
        live_all: dict[str, float],
        exposure: float,
    ) -> dict[str, float]:
        """Select top-K SP500 stocks by 63d momentum above 200d SMA."""
        prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
        if len(prices_window) < MOMENTUM_WINDOW:
            return {"IEF": exposure}

        exclude = {"TLT", "IEF", "SPY"}
        live = {s: p for s, p in live_all.items() if s not in exclude}

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
            return {"IEF": exposure}

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
            return {"IEF": exposure}

        per_weight = exposure / len(selected)
        return {sym: per_weight for sym in selected}


NAME = "breadth_gated_sp500"
HYPOTHESIS = (
    "SP500 market breadth momentum gate: compute percent of SP500 stocks above 50d SMA "
    "(breadth ratio); when breadth > 60th percentile of trailing 252d distribution "
    "(broad participation) hold top-15 SP500 stocks by 63d momentum; when breadth < "
    "40th percentile (deteriorating breadth) hold IEF 60%+TLT 37%; SPY 200d bear override "
    "to full TLT; biweekly rebalance"
)

STRATEGY = BreadthGatedSP500()
