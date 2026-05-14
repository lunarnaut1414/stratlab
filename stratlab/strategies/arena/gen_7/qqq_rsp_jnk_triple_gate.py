"""QQQ RSP+JNK Triple Gate — gen_7 sonnet-7 (attempt 10)

Hypothesis: A 3-state regime using RSP vs SPY breadth AND JNK 30d MA:
- Both bullish (RSP>SPY 30d return AND JNK above 30d SMA): QQQ 97%
- JNK bullish only (credit strong but breadth weak): SPY 97%
- Neither: TLT 60% + SHY 37%

Simplification from attempt 9: remove the 4th tier and just use JNK MA
(not acceleration) as the credit signal. The gen6_opus3_ensemble showed
that RSP breadth + JNK credit + smallcap signals work well together.
This is a simpler, cleaner 3-tier version.

SPY 200d SMA outer bear gate.
Weekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "SHY", "RSP", "JNK"]

REBALANCE_EVERY = 5        # weekly
BREADTH_WINDOW = 30        # RSP vs SPY 30d return
JNK_MA = 30                # JNK 30d SMA
TREND_WINDOW = 200
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_SHY = "SHY"
_RSP = "RSP"
_JNK = "JNK"


class QQQRspJnkTripleGate(Strategy):
    """RSP breadth + JNK MA dual gate -> QQQ/SPY/TLT-SHY 3-tier."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_window: int = BREADTH_WINDOW,
        jnk_ma: int = JNK_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            jnk_ma=jnk_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.jnk_ma = int(jnk_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.trend_window, self.breadth_window) + 10
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

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            # Bear: TLT+SHY
            if _TLT in live:
                target[_TLT] = 0.60 * self.exposure
            if _SHY in live:
                target[_SHY] = 0.37 * self.exposure
        else:
            need = max(self.jnk_ma, self.breadth_window) + 5
            prices = ctx.closes_window(need)

            # Signal 1: RSP vs SPY breadth (30d return)
            breadth_bull = False
            if _RSP in prices.columns and _SPY in prices.columns:
                rsp_col = prices[_RSP].dropna()
                spy_col = prices[_SPY].dropna()
                min_len = min(len(rsp_col), len(spy_col))
                if min_len >= self.breadth_window:
                    rsp_ret = float(rsp_col.iloc[-1] / rsp_col.iloc[-self.breadth_window] - 1.0)
                    spy_ret = float(spy_col.iloc[-1] / spy_col.iloc[-self.breadth_window] - 1.0)
                    breadth_bull = (np.isfinite(rsp_ret) and np.isfinite(spy_ret) and
                                    rsp_ret > spy_ret)

            # Signal 2: JNK above 30d SMA (credit)
            credit_bull = False
            if _JNK in prices.columns:
                jnk_col = prices[_JNK].dropna()
                if len(jnk_col) >= self.jnk_ma:
                    jnk_now = float(jnk_col.iloc[-1])
                    jnk_sma = float(jnk_col.iloc[-self.jnk_ma:].mean())
                    credit_bull = jnk_now > jnk_sma

            if breadth_bull and credit_bull:
                # Both bullish: QQQ
                if _QQQ in live:
                    target[_QQQ] = self.exposure
            elif credit_bull:
                # Credit strong only: SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Credit weak: TLT+SHY
                if _TLT in live:
                    target[_TLT] = 0.60 * self.exposure
                if _SHY in live:
                    target[_SHY] = 0.37 * self.exposure

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


NAME = "qqq_rsp_jnk_triple_gate"
HYPOTHESIS = (
    "QQQ RSP-breadth + JNK-MA triple gate: RSP>SPY 30d return (broad breadth) AND "
    "JNK above 30d SMA (credit) -> QQQ 97%; JNK only -> SPY 97%; neither -> "
    "TLT 60%+SHY 37%; SPY 200d SMA bear outer gate; weekly rebalance; "
    "3-tier composite of independent breadth and credit signals"
)

STRATEGY = QQQRspJnkTripleGate()
