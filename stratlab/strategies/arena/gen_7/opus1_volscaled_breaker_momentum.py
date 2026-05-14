"""SP500 momentum with volatility-scaled circuit breaker — opus-1 gen_7

Mutation of gen7_momentum_drawdown_breaker (parent IS Calmar 0.93).

Parent: top-15 by 42d momentum, fixed 7% drawdown circuit-breaker on portfolio
21-bar rolling peak; 10-bar cooldown.
Mutation: replace fixed 7% threshold with VOLATILITY-SCALED depth equal to
1.5x trailing 60d realized vol (annualized → bar-vol then 21-bar). Floor at
5%, cap at 12%. Cooldown shortened to 5 bars to re-enter sooner.

Rationale: a fixed 7% circuit triggers too easily in calm regimes (false
positives) and too late in storms (true peak draw exceeds 7% before exit).
Scaling the trigger to recent realized vol gives a more regime-aware breaker.
The cooldown shortening compensates for spurious calm-regime trips.

Different mechanism (vol-scaled vs fixed) and different cooldown — should
clear corr filter vs parent.
"""
from __future__ import annotations

import math

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 42
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
PEAK_WINDOW = 21
COOLDOWN_BARS = 5
RV_WINDOW = 60
RV_MULT = 1.5
DD_FLOOR = 0.05
DD_CEIL = 0.12


class VolScaledBreakerMomentum(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        peak_window: int = PEAK_WINDOW,
        cooldown_bars: int = COOLDOWN_BARS,
        rv_window: int = RV_WINDOW,
        rv_mult: float = RV_MULT,
        dd_floor: float = DD_FLOOR,
        dd_ceil: float = DD_CEIL,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            peak_window=peak_window,
            cooldown_bars=cooldown_bars,
            rv_window=rv_window,
            rv_mult=rv_mult,
            dd_floor=dd_floor,
            dd_ceil=dd_ceil,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.peak_window = int(peak_window)
        self.cooldown_bars = int(cooldown_bars)
        self.rv_window = int(rv_window)
        self.rv_mult = float(rv_mult)
        self.dd_floor = float(dd_floor)
        self.dd_ceil = float(dd_ceil)

        self._equity_history: list[float] = []
        self._cooldown_remaining: int = 0
        self._circuit_open: bool = False

    def on_start(self) -> None:
        self._equity_history = []
        self._cooldown_remaining = 0
        self._circuit_open = False

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.rv_window) + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Track equity history for portfolio-DD
        self._equity_history.append(equity)
        max_keep = max(self.peak_window, self.rv_window) + 5
        if len(self._equity_history) > max_keep:
            self._equity_history = self._equity_history[-max_keep:]

        # Compute volatility-scaled DD threshold from SPY 60d realized vol
        # (use SPY bar returns; vol_21bar = annualized_vol * sqrt(21/252))
        dd_threshold = 0.07  # default if signal unavailable
        try:
            spy_hist = ctx.history("SPY")
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.rv_window + 1:
                spy_arr = spy_close.values[-(self.rv_window + 1):]
                rets = np.diff(np.log(spy_arr))
                if len(rets) >= 10 and np.all(np.isfinite(rets)):
                    daily_vol = float(np.std(rets, ddof=1))
                    # Convert daily-bar vol to ~21-bar (peak window) horizon vol
                    horizon_vol = daily_vol * math.sqrt(self.peak_window)
                    candidate = self.rv_mult * horizon_vol
                    if np.isfinite(candidate):
                        dd_threshold = max(self.dd_floor, min(self.dd_ceil, candidate))
        except Exception:
            pass

        # Drawdown from rolling peak
        circuit_triggered = False
        if len(self._equity_history) >= self.peak_window:
            recent = self._equity_history[-self.peak_window:]
            peak = max(recent)
            if peak > 0:
                current_dd = (peak - equity) / peak
                if current_dd >= dd_threshold:
                    circuit_triggered = True

        if circuit_triggered:
            self._circuit_open = True
            self._cooldown_remaining = self.cooldown_bars
        elif self._circuit_open:
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            else:
                self._circuit_open = False

        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend
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
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
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
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
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


NAME = "opus1_volscaled_breaker_momentum"
HYPOTHESIS = (
    "Momentum with deep-circuit-breaker volatility-scaled trigger: top-15 SP500 by 42d "
    "momentum but circuit-breaker depth is 1.5x trailing 60d realized-volatility (adaptive, "
    "not fixed 7%); minimum 5%, maximum 12%; cooldown shortened to 5 bars from 10; trigger "
    "scales with regime volatility instead of fixed level"
)

UNIVERSE = _universe

STRATEGY = VolScaledBreakerMomentum()
