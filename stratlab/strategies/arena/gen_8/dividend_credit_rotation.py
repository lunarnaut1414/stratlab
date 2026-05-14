"""Dividend-Growth Credit Rotation — gen_8 sonnet-8

Hypothesis: Hold equally-weighted dividend income ETFs (DVY + VIG) when
credit is healthy (JNK > 30d SMA) AND SPY is above 100d SMA (dual gate);
rotate to TLT 97% when either gate fails. Weekly rebalance.

Rationale: Dividend-oriented ETFs (DVY = iShares Dividend, VIG = Vanguard
Dividend Growth) are not represented in the leaderboard — existing strategies
either hold growth QQQ or individual SP500 stocks. Dividend stocks have lower
beta and higher quality characteristics, producing a different return profile
than momentum strategies. The dual credit+trend gate from gen6_hy_credit_qqq
is proven OOS-robust (OOS Calmar 0.10, but its QQQ loading already exists);
routing the same signal to dividend ETFs adds a distinct factor tilt.

Key distinction from gen6_hy_credit_qqq_rotation (holds QQQ, corr ~0.4):
- DVY+VIG is value/dividend factor, QQQ is growth/momentum → different beta
- Dividend ETFs have lower IS-window CAGR but also lower drawdown → Calmar
  ratio comparable despite lower absolute returns
- Loss mode behavior: dividend stocks tend to outperform growth in credit
  stress regimes (the "flight to quality within equities" effect)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
REBALANCE_EVERY = 5           # weekly
JNK_MA = 30                   # credit signal window
SPY_TREND = 100               # market trend window
EXPOSURE = 0.97
_JNK = "JNK"
_SPY = "SPY"
_TLT = "TLT"
_DIVS = ["DVY", "VIG"]        # dividend income ETFs


class DividendCreditRotation(Strategy):
    """Dual-gate (JNK credit + SPY trend) routing to dividend vs TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        spy_trend: int = SPY_TREND,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            spy_trend=spy_trend,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.spy_trend = int(spy_trend)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.spy_trend) + 5
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

        # ---- JNK credit gate ----
        credit_ok = False
        try:
            jnk_hist = ctx.history(_JNK)
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma:
                jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_now = float(jnk_close.iloc[-1])
                credit_ok = jnk_now > jnk_ma_val
        except (KeyError, Exception):
            credit_ok = False

        # ---- SPY trend gate ----
        trend_ok = False
        try:
            spy_hist = ctx.history(_SPY)
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.spy_trend:
                spy_ma_val = float(spy_close.iloc[-self.spy_trend:].mean())
                spy_now = float(spy_close.iloc[-1])
                trend_ok = spy_now > spy_ma_val
        except (KeyError, Exception):
            trend_ok = False

        target: dict[str, float] = {}

        if credit_ok and trend_ok:
            # Risk-on: equal-weight dividend ETFs
            available = [s for s in _DIVS if s in live]
            if available:
                per_weight = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_weight
            else:
                # Fallback to TLT if neither dividend ETF has data
                if _TLT in live:
                    target[_TLT] = self.exposure
        else:
            # Risk-off: TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        # ---- Build orders ----
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


UNIVERSE = ["DVY", "VIG", "TLT", "JNK", "SPY"]

NAME = "dividend_credit_rotation"
HYPOTHESIS = (
    "Dividend-growth ETF rotation: hold equally-weighted DVY+VIG (dividend income ETFs) "
    "when JNK above 30d SMA AND SPY above 100d SMA (credit+trend dual gate); "
    "rotate to TLT 97% when credit or trend fails; weekly rebalance; income-tilt rotation "
    "not represented in leaderboard"
)

STRATEGY = DividendCreditRotation()
