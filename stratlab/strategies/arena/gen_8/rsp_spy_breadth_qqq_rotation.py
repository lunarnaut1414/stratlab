"""RSP-SPY Breadth Signal to QQQ/TLT Rotation — gen_8 sonnet-10

Hypothesis: When RSP (Invesco Equal-Weight S&P 500) outperforms SPY on a
42-day basis, it signals broad market participation where most stocks are
contributing — this is a high-breadth, risk-on regime. In this regime, hold
QQQ (concentrated growth/tech leadership works in broad participation).

When SPY outperforms RSP, it signals narrow, mega-cap-led market — fewer stocks
are driving the index. This is a warning sign; hold SPY 60% + IEF 37%.

When SPY is below its 200d SMA (bear), hold TLT 97%.

Rationale: The RSP/SPY relative performance ratio is a breadth signal based on
market structure (equal-weight vs cap-weight differential). It's distinct from:
- VIX level gates (volatility-based)
- Credit spread gates (JNK/LQD, HYG)
- Yield curve signals (TNX, slope)
- Sector ETF breadth count (crosses sector boundaries)

This signal was used in historical breadth analysis (RSP/SPY divergence as a
leading indicator for market corrections). When RSP leads, the broad market
momentum favors growth positioning.

Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
BREADTH_WINDOW = 42       # 42-day RSP vs SPY return comparison
SPY_TREND_WINDOW = 200    # SPY bear gate
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_IEF = "IEF"
_TLT = "TLT"
_RSP = "RSP"


class RSPSPYBreadthQQQRotation(Strategy):
    """RSP-SPY breadth signal to QQQ/SPY+IEF/TLT rotation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_window: int = BREADTH_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.breadth_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY bear gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # RSP vs SPY 42-day relative performance
        broad_participation = True  # default if signal unavailable
        try:
            rsp_hist = ctx.history(_RSP)
            if (rsp_hist is not None
                    and len(rsp_hist) >= self.breadth_window + 2
                    and len(spy_hist) >= self.breadth_window + 2):
                rsp_close = rsp_hist["close"].dropna()
                if (len(rsp_close) >= self.breadth_window + 1
                        and len(spy_close) >= self.breadth_window + 1):
                    rsp_ret_42 = float(
                        rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window - 1] - 1.0
                    )
                    spy_ret_42 = float(
                        spy_close.iloc[-1] / spy_close.iloc[-self.breadth_window - 1] - 1.0
                    )
                    # RSP outperforms SPY = broad participation
                    broad_participation = rsp_ret_42 > spy_ret_42
        except (KeyError, Exception):
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
            # Bear: TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        elif broad_participation:
            # Broad bull: QQQ
            if _QQQ in live:
                target[_QQQ] = self.exposure

        else:
            # Narrow leadership (SPY leads RSP): SPY + IEF blend
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _IEF in live:
                target[_IEF] = self.exposure * 0.382

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


UNIVERSE = [_RSP, _SPY, _QQQ, _IEF, _TLT]

NAME = "rsp_spy_breadth_qqq_rotation"
HYPOTHESIS = (
    "RSP-SPY breadth rotation signal to QQQ/TLT: when RSP (equal-weight SP500) 42d return "
    "exceeds SPY 42d return (broad market participation, risk-on) hold QQQ 97%; "
    "when SPY leads RSP (narrowing leadership) hold SPY 60%+IEF 37%; "
    "when SPY below 200d SMA hold TLT 97%; biweekly rebalance; "
    "market-breadth-as-signal distinct from sector breadth count"
)

STRATEGY = RSPSPYBreadthQQQRotation()
