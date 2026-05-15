"""gen_9 opus-1 — Character-Adaptive ETF Rotation with EFA vs SPY signal.

Parent: gen9_dvy_spy_character_adaptive (IS Calmar 1.24, h1=1.69 h2=0.84 H1-DOMINANT).
Mutation:
  - Replace DVY (US dividend ETF) with EFA (developed-intl ETF) as character signal.
  - REPLACE stock-selection with FACTOR-ETF ROTATION (VTV value vs VUG growth):
      * When EFA leads SPY 63d (intl risk-on, USD weak): VTV (value) tilted —
        intl outperformance historically coincides with value factor strength
        (financials, energy, materials) and USD weakness.
      * When SPY leads EFA (US dominance, USD strong): VUG (growth) tilted —
        US dominance is usually tech/growth-led.
  - SPY 200d outer bear gate -> TLT.
  - Goal: preserve the criterion-adaptation thesis but use factor ETFs to
    AVOID the SP500-xsect-momentum corr attractor (the original DVY parent
    had corr 0.84; the first attempt with EFA + stock selection had corr 0.88).
  - Same biweekly rebalance, exposure 97%.

This is structurally distinct from any leaderboard entry: no other strategy
uses EFA-vs-SPY as a value-vs-growth ETF rotation signal.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
TREND_WINDOW = 200
CHARACTER_WINDOW = 63
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_EFA = "EFA"
_VTV = "VTV"
_VUG = "VUG"


class Opus1EfaSpyCharacterAdaptive(Strategy):
    """EFA vs SPY character drives VTV (value) vs VUG (growth) ETF tilt."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, CHARACTER_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < TREND_WINDOW + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < TREND_WINDOW:
            return []
        spy_sma = float(spy_close.iloc[-TREND_WINDOW:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # EFA vs SPY 63d character signal
        intl_regime = False
        try:
            efa_hist = ctx.history(_EFA)
            if efa_hist is not None and len(efa_hist) >= CHARACTER_WINDOW + 2:
                efa_close = efa_hist["close"].dropna()
                if len(efa_close) >= CHARACTER_WINDOW + 1:
                    efa_ret = float(
                        efa_close.iloc[-1] / efa_close.iloc[-CHARACTER_WINDOW] - 1.0
                    )
                    spy_ret_char = float(
                        spy_close.iloc[-1] / spy_close.iloc[-CHARACTER_WINDOW] - 1.0
                    )
                    intl_regime = (
                        np.isfinite(efa_ret) and np.isfinite(spy_ret_char)
                        and efa_ret >= spy_ret_char
                    )
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = EXPOSURE
        else:
            if intl_regime:
                # Intl leads -> value factor (VTV)
                if _VTV in live:
                    target[_VTV] = EXPOSURE
                elif _SPY in live:
                    target[_SPY] = EXPOSURE
            else:
                # US leads -> growth factor (VUG)
                if _VUG in live:
                    target[_VUG] = EXPOSURE
                elif _SPY in live:
                    target[_SPY] = EXPOSURE

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


UNIVERSE = ["SPY", "TLT", "EFA", "VTV", "VUG"]

NAME = "opus1_efa_spy_character_adaptive"
HYPOTHESIS = (
    "Mutate gen9_dvy_spy_character_adaptive: use EFA vs SPY 63d as character signal "
    "driving VTV (value) vs VUG (growth) factor ETF rotation; EFA leads SPY -> VTV; "
    "SPY leads EFA -> VUG; SPY 200d bear gate -> TLT; ETF-only avoids SP500-xsect-mom "
    "corr attractor; biweekly rebalance."
)

STRATEGY = Opus1EfaSpyCharacterAdaptive()
