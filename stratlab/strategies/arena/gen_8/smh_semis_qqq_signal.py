"""Semiconductor Cycle as QQQ/SPY Signal — gen_8 sonnet-2

Hypothesis: Use SMH (semiconductor ETF) vs QQQ relative 42d momentum as a
tech-cycle leading indicator. When semis outperform QQQ (semiconductors leading
broad tech = early-cycle growth), hold QQQ at 97%. When semis significantly
underperform QQQ (or neutral), hold SPY at 80%. TLT when SPY < 200d SMA (bear).

Rationale:
- Semiconductors are a leading indicator for the tech cycle: they lead QQQ
  when the cycle is expanding (capacity orders precede shipments precede software
  demand). SMH outperformance signals durable tech momentum.
- Routes exposure through QQQ and SPY (not SMH directly) — SMH is purely a signal
- Distinct from gen5_semi_cycle_smh: that strategy held SMH directly;
  this strategy uses SMH as signal and routes to QQQ/SPY/TLT
- Distinct from all existing credit/VIX/yield regime strategies: uses
  semiconductor cycle as regime indicator

Rebalance: weekly (every 5 bars)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 42           # 42d for SMH vs QQQ comparison
TREND_WINDOW = 200        # SPY 200d SMA
EXPOSURE = 0.97
SPY_MID_EXPOSURE = 0.80   # when semis neutral/lagging: hold SPY at lower exposure
_SMH = "SMH"
_QQQ = "QQQ"
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"


class SMHSemisQQQSignal(Strategy):
    """SMH semiconductor relative momentum as QQQ/SPY/TLT regime signal."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
        spy_mid_exposure: float = SPY_MID_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            trend_window=trend_window,
            exposure=exposure,
            spy_mid_exposure=spy_mid_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self.spy_mid_exposure = float(spy_mid_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT defensive
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute 42d returns for SMH and QQQ
            need = self.mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_window:
                return []

            smh_ret = None
            if _SMH in prices.columns:
                col = prices[_SMH].dropna()
                if len(col) >= self.mom_window + 1:
                    smh_ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)

            qqq_ret = None
            if _QQQ in prices.columns:
                col = prices[_QQQ].dropna()
                if len(col) >= self.mom_window + 1:
                    qqq_ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)

            # Determine regime from SMH vs QQQ spread
            if smh_ret is not None and qqq_ret is not None:
                spread = smh_ret - qqq_ret
                if spread > 0.0:
                    # Semis leading tech — strong growth regime: hold QQQ
                    if _QQQ in live:
                        target[_QQQ] = self.exposure
                else:
                    # Semis lagging — SPY at lower exposure
                    if _SPY in live:
                        target[_SPY] = self.spy_mid_exposure
            elif qqq_ret is not None:
                # SMH data missing: default to QQQ
                if _QQQ in live:
                    target[_QQQ] = self.spy_mid_exposure
            else:
                # Neither available: SPY
                if _SPY in live:
                    target[_SPY] = self.spy_mid_exposure

        orders: list[Order] = []

        # Exit positions not in target
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


UNIVERSE = [_SMH, _QQQ, _SPY, _TLT, _IEF]

NAME = "smh_semis_qqq_signal"
HYPOTHESIS = (
    "SMH semiconductor 42d return vs QQQ as tech-cycle signal: when SMH outperforms "
    "QQQ (semis leading) hold QQQ 97%; when semis lag hold SPY 80%; TLT in bear "
    "(SPY < 200d SMA); weekly rebalance; semiconductor cycle as regime indicator "
    "routing to QQQ/SPY (not holding SMH directly)"
)

STRATEGY = SMHSemisQQQSignal()
