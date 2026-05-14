"""SP500 momentum with max-drawdown circuit breaker.

Hypothesis: hold top-15 SP500 stocks by 42d momentum when SPY above 200d SMA AND
portfolio has not dropped >7% from 21d peak; rotate to TLT when circuit triggers;
rebalance every 10 bars; drawdown-aware momentum reduces max_dd vs pure momentum.

Rationale: cross-sectional momentum strategies tend to have large max drawdowns during
regime breaks (e.g., 2015-16, 2018 Q4). Adding a portfolio-level circuit breaker that
monitors the 21-bar rolling peak and exits to TLT when drawdown exceeds 7% should
reduce max drawdown without significantly degrading CAGR. This is different from
VIX-level gates (reactive to external signal) — it's reactive to the portfolio itself.

Distinction from existing strategies:
  - Portfolio-level drawdown circuit breaker (not VIX or credit signal)
  - Uses 21-bar rolling peak of portfolio value as reference
  - 7% drawdown threshold forces rotation to TLT
  - SPY 200d SMA gate as secondary filter
  - Once in TLT defensive, requires 10-bar cooldown before re-entering equity momentum
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 42       # ~2 months
SPY_TREND_WINDOW = 200     # SPY 200d SMA
TOP_K = 15
EXPOSURE = 0.97
DD_THRESHOLD = 0.07        # 7% drawdown from 21-bar peak triggers circuit
PEAK_WINDOW = 21           # bars for rolling peak
COOLDOWN_BARS = 10         # bars to stay in TLT after circuit trigger


class MomentumDrawdownBreaker(Strategy):
    """Top-15 SP500 by 42d momentum; SPY 200d SMA gate + portfolio drawdown circuit breaker."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        dd_threshold: float = DD_THRESHOLD,
        peak_window: int = PEAK_WINDOW,
        cooldown_bars: int = COOLDOWN_BARS,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            dd_threshold=dd_threshold,
            peak_window=peak_window,
            cooldown_bars=cooldown_bars,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.dd_threshold = float(dd_threshold)
        self.peak_window = int(peak_window)
        self.cooldown_bars = int(cooldown_bars)

        # Internal state
        self._equity_history: list[float] = []
        self._cooldown_remaining: int = 0
        self._circuit_open: bool = False

    def on_start(self) -> None:
        self._equity_history = []
        self._cooldown_remaining = 0
        self._circuit_open = False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Track equity for drawdown monitoring
        self._equity_history.append(equity)
        # Keep only the last peak_window + 5 values to avoid unbounded memory
        if len(self._equity_history) > self.peak_window + 5:
            self._equity_history = self._equity_history[-(self.peak_window + 5):]

        # Calculate drawdown from rolling peak
        circuit_triggered = False
        if len(self._equity_history) >= self.peak_window:
            recent = self._equity_history[-self.peak_window:]
            peak = max(recent)
            if peak > 0:
                current_dd = (peak - equity) / peak
                if current_dd >= self.dd_threshold:
                    circuit_triggered = True

        # Update circuit breaker state
        if circuit_triggered:
            self._circuit_open = True
            self._cooldown_remaining = self.cooldown_bars
        elif self._circuit_open:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            else:
                self._circuit_open = False

        # Check if this is a rebalance bar
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

        target: dict[str, float] = {}

        if self._circuit_open or not spy_bull:
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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "momentum_drawdown_breaker"
HYPOTHESIS = (
    "SP500 momentum with max-drawdown circuit breaker: hold top-15 SP500 stocks by 42d momentum "
    "when SPY above 200d SMA AND portfolio has not dropped >7% from 21d peak; rotate to TLT when "
    "circuit triggers; rebalance every 10 bars; drawdown-aware momentum reduces max_dd relative to pure momentum"
)

UNIVERSE = _universe

STRATEGY = MomentumDrawdownBreaker()
