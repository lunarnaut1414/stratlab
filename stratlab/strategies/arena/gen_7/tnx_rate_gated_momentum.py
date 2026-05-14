"""SP500 top-20 momentum gated by TNX yield direction.

Hypothesis: hold top-20 SP500 stocks by 63d return when SPY above 200d SMA AND
TNX (10Y yield) 20d MA is falling or flat (falling-rates regime favors growth);
rotate to TLT 97% when rates rising (TNX 20d MA > 60d MA); biweekly rebalance.

Rationale: Prior TNX-gated strategies on the leaderboard failed because they routed
to narrow sector ETFs (XLF, XLU) instead of individual stocks. This strategy uses
TNX as a *secondary* regime filter alongside SPY 200d SMA (the primary market trend
gate), then applies cross-sectional stock momentum in the risk-on window. The IS
window (2010-2018) includes both falling-rate (2010-2015) and rising-rate (2015-2018)
periods, making this a meaningful regime filter.

Distinction from existing strategies:
  - TNX 20d vs 60d MA used as rate-direction filter (not just yield level)
  - Routes to individual SP500 stocks (not sector ETFs) in risk-on
  - Dual gate: SPY 200d SMA AND TNX falling-rate condition
  - TLT defensive when rates rising (duration benefits from falling-rate environment)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # ~3 months
SPY_TREND_WINDOW = 200     # SPY 200d SMA
TNX_FAST_MA = 20           # fast TNX MA
TNX_SLOW_MA = 60           # slow TNX MA
TOP_K = 20
EXPOSURE = 0.97
_TNX = "^TNX"


class TnxRateGatedMomentum(Strategy):
    """Top-20 SP500 by 63d momentum; SPY 200d SMA gate + TNX falling-rate gate; TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        tnx_fast_ma: int = TNX_FAST_MA,
        tnx_slow_ma: int = TNX_SLOW_MA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            tnx_fast_ma=tnx_fast_ma,
            tnx_slow_ma=tnx_slow_ma,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.tnx_fast_ma = int(tnx_fast_ma)
        self.tnx_slow_ma = int(tnx_slow_ma)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.tnx_slow_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # Check TNX direction: falling rates = risk-on for growth stocks
        tnx_rates_falling = True  # default to risk-on if TNX unavailable
        try:
            tnx_hist = ctx.history(_TNX)
            if tnx_hist is not None and len(tnx_hist) >= self.tnx_slow_ma + 2:
                tnx_close = tnx_hist["close"].dropna()
                if len(tnx_close) >= self.tnx_slow_ma:
                    tnx_fast = float(tnx_close.iloc[-self.tnx_fast_ma:].mean())
                    tnx_slow = float(tnx_close.iloc[-self.tnx_slow_ma:].mean())
                    # Rates rising = risk-off (TNX fast MA > slow MA)
                    tnx_rates_falling = tnx_fast <= tnx_slow
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        risk_on = spy_bull and tnx_rates_falling

        if not risk_on:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K momentum stocks
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / k
                for sym in ranked:
                    target[sym] = per_weight

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", _TNX]


NAME = "tnx_rate_gated_momentum"
HYPOTHESIS = (
    "SP500 top-20 momentum gated by TNX yield direction: hold top-20 SP500 stocks by 63d return "
    "when SPY above 200d SMA AND TNX (10Y yield) 20d MA is falling or flat (falling-rates regime "
    "favors growth); rotate to TLT 97% when rates rising (TNX 20d MA > 60d MA); biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = TnxRateGatedMomentum()
