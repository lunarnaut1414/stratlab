"""Market Breadth Proxy Rotation (RSP vs SPY) — gen_5 sonnet-7

Hypothesis: When equal-weight S&P 500 (RSP) outperforms cap-weight SPY on a
rolling basis, market participation is broad — a positive breadth signal that
suggests durable bull market conditions. Concentration in the strongest sector
(QQQ/tech) is then appropriate. When SPY outperforms RSP (narrow leadership,
mega-cap driven), risk is building, and a more balanced or defensive posture
is warranted.

Signal (biweekly check):
  - RSP/SPY ratio relative to its 40-day MA:
    - ratio > MA (broad participation): hold QQQ (aggressive/growth)
    - ratio < MA (narrow/mega-cap): hold SPY (moderate)
    - ratio < MA AND SPY below 200d MA: hold TLT (defensive)

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "TLT", "RSP"]

_BREADTH_MA = 40      # RSP/SPY ratio MA
_TREND_WINDOW = 200   # SPY 200d MA for defensive gate
_MIN_HISTORY = max(_BREADTH_MA, _TREND_WINDOW) + 5
_REBALANCE = 10       # biweekly
_EXPOSURE = 0.97


class BreadthProxyRotation(Strategy):
    """RSP/SPY ratio as breadth proxy: QQQ in broad mkt, SPY in narrow, TLT in bear."""

    def __init__(
        self,
        breadth_ma: int = _BREADTH_MA,
        trend_window: int = _TREND_WINDOW,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            breadth_ma=breadth_ma,
            trend_window=trend_window,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.breadth_ma = breadth_ma
        self.trend_window = trend_window
        self.rebalance = rebalance
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < _MIN_HISTORY:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        try:
            rsp_hist = ctx.history("RSP")
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []

        needed = max(self.breadth_ma, self.trend_window) + 1
        if len(rsp_hist) < needed or len(spy_hist) < needed:
            return []

        spy_closes = spy_hist["close"].values
        rsp_closes = rsp_hist["close"].values

        # RSP/SPY ratio and its breadth_ma-day moving average
        min_len = min(len(rsp_closes), len(spy_closes))
        if min_len < self.breadth_ma + 1:
            return []

        ratio = rsp_closes[-min_len:] / spy_closes[-min_len:]
        current_ratio = float(ratio[-1])
        ratio_ma = float(np.mean(ratio[-self.breadth_ma:]))

        # SPY trend: above 200d MA?
        spy_ma200 = float(np.mean(spy_closes[-self.trend_window:]))
        spy_uptrend = float(spy_closes[-1]) > spy_ma200

        # Determine regime
        breadth_positive = current_ratio > ratio_ma

        if not spy_uptrend:
            # Bear market: hold TLT
            target_sym = "TLT"
        elif breadth_positive:
            # Broad market participation: QQQ
            target_sym = "QQQ"
        else:
            # Narrow leadership (mega-cap only): SPY
            target_sym = "SPY"

        all_etfs = ["SPY", "QQQ", "TLT"]
        others = [s for s in all_etfs if s != target_sym]

        closes = ctx.closes()
        if target_sym not in closes or closes[target_sym] <= 0:
            return []

        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit other positions
        for other_sym in others:
            other_pos = ctx.position(other_sym)
            if other_pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=other_pos.size, symbol=other_sym))

        # Size target position
        target_pos = ctx.position(target_sym)
        target_price = float(closes[target_sym])
        target_shares = int(equity * self.exposure / target_price)
        delta = target_shares - int(target_pos.size)
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))
        elif delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=target_sym))

        return orders


NAME = "atr_momentum_etf"
HYPOTHESIS = (
    "Market breadth proxy rotation using RSP/SPY ratio: hold QQQ when RSP outperforms "
    "SPY on 40d MA (broad participation = growth regime), SPY when narrow leadership "
    "(mega-cap only), TLT when SPY below 200d MA (bear). RSP/SPY breadth signal is novel "
    "vs all existing leaderboard strategies. Biweekly rebalance."
)

STRATEGY = BreadthProxyRotation(
    breadth_ma=_BREADTH_MA,
    trend_window=_TREND_WINDOW,
    rebalance=_REBALANCE,
    exposure=_EXPOSURE,
)
