"""opus-1 / gen_8 — Dividend ETF Rotation with Rate-Trend Gate

Mutation of gen8_dividend_credit_rotation (IS Calmar 0.59, corr 0.48 LOWEST).

Parent: equal-weight DVY+VIG when JNK > 30d SMA AND SPY > 100d SMA;
else TLT. Weekly rebalance.

This variant preserves the parent's low-correlation property (which comes from
the unusual dividend-factor branch) and adds a *rate-trend* gate as a third
signal — TNX 60d slope. The hypothesis: dividend stocks are bond-like; they
outperform when long rates are FALLING (TNX 60d slope < 0) and underperform
when rates are RISING. So we hold DVY+VIG (equal-weight) only when:
  - JNK > 30d SMA (credit healthy)  AND
  - SPY > 100d SMA (trend healthy)  AND
  - TNX 20d MA < 60d MA (rates trending down)

When JNK or SPY gate fails: TLT 97% (parent behavior).
When only TNX gate fails (rates rising but credit+trend healthy): SPY 97%
(intermediate state — still bullish but dividend factor not favored).

Same DVY+VIG core (both pre-2010 inception, no factor-ETF cache gap risk).
Weekly rebalance preserved.

Goal: lift Calmar from 0.59 by adding rate-trend selectivity, while preserving
the corr=0.48 advantage.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
JNK_MA = 30
SPY_TREND = 100
TNX_FAST = 20
TNX_SLOW = 60
EXPOSURE = 0.97
_JNK = "JNK"
_SPY = "SPY"
_TLT = "TLT"
_TNX = "^TNX"
_DIVS = ["DVY", "VIG"]


class DividendRateCreditRotation(Strategy):
    """Triple-gate (credit + trend + rate-direction) routing to dividend ETFs."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        spy_trend: int = SPY_TREND,
        tnx_fast: int = TNX_FAST,
        tnx_slow: int = TNX_SLOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            spy_trend=spy_trend,
            tnx_fast=tnx_fast,
            tnx_slow=tnx_slow,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.spy_trend = int(spy_trend)
        self.tnx_fast = int(tnx_fast)
        self.tnx_slow = int(tnx_slow)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.spy_trend, self.tnx_slow) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        credit_ok = False
        try:
            jnk_hist = ctx.history(_JNK)
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma:
                credit_ok = float(jnk_close.iloc[-1]) > float(
                    jnk_close.iloc[-self.jnk_ma:].mean()
                )
        except Exception:
            pass

        trend_ok = False
        try:
            spy_hist = ctx.history(_SPY)
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.spy_trend:
                trend_ok = float(spy_close.iloc[-1]) > float(
                    spy_close.iloc[-self.spy_trend:].mean()
                )
        except Exception:
            pass

        # Rates trending DOWN (favorable for dividend stocks): TNX 20d MA < 60d MA
        rates_falling = False
        try:
            tnx_hist = ctx.history(_TNX)
            if tnx_hist is not None:
                tnx_close = tnx_hist["close"].dropna()
                if len(tnx_close) >= self.tnx_slow:
                    fast = float(tnx_close.iloc[-self.tnx_fast:].mean())
                    slow = float(tnx_close.iloc[-self.tnx_slow:].mean())
                    rates_falling = fast < slow
        except Exception:
            pass

        target: dict[str, float] = {}

        if credit_ok and trend_ok and rates_falling:
            # All gates aligned: dividend factor regime
            available = [s for s in _DIVS if s in live]
            if available:
                per_w = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_w
            elif _TLT in live:
                target[_TLT] = self.exposure
        elif credit_ok and trend_ok and not rates_falling:
            # Credit + trend OK but rates rising — neutral SPY
            if _SPY in live:
                target[_SPY] = self.exposure
        else:
            # Credit or trend failed: defensive TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
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


UNIVERSE = ["DVY", "VIG", "TLT", "JNK", "SPY", "^TNX"]

NAME = "opus1_dividend_rate_credit_rotation"
HYPOTHESIS = (
    "Mutation of dividend_credit_rotation: add TNX 20d<60d rate-trend gate as 3rd signal. "
    "Hold DVY+VIG equal-weight when JNK>30d-SMA AND SPY>100d-SMA AND rates falling; "
    "fall back to SPY when only rates rising; TLT when credit/trend fails; weekly rebalance; "
    "preserve parent low-corr (0.48) property"
)

STRATEGY = DividendRateCreditRotation()
