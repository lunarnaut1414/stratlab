"""Credit Spread JNK/LQD Timing Strategy — gen_5 sonnet-2

Hypothesis: The ratio of JNK (SPDR high yield) to LQD (iShares IG corporate)
captures credit spread dynamics. When spreads tighten (JNK outperforms LQD,
ratio rising), risk appetite is high and high-yield bonds outperform. When
spreads widen (ratio falling), defensive IG bonds are preferred.

Signal: 20-day MA vs 60-day MA on the JNK/LQD price ratio.
- When 20d MA > 60d MA (ratio trending up, tightening spreads): hold JNK
- When 20d MA < 60d MA (ratio trending down, widening spreads): hold LQD

Both JNK (2007-12-04) and LQD (2002-07-30) have data well before IS start.

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy


# Both start before IS window (JNK: 2007-12-04, LQD: 2002-07-30)
UNIVERSE = ["JNK", "LQD"]

_FAST_MA = 20
_SLOW_MA = 90
_MIN_HISTORY = _SLOW_MA + 5
_REBALANCE = 5  # weekly check
_EXPOSURE = 0.97


class CreditSpreadHygLqd(Strategy):
    """JNK/LQD ratio MA crossover for credit spread regime timing."""

    def __init__(
        self,
        fast_ma: int = _FAST_MA,
        slow_ma: int = _SLOW_MA,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rebalance = rebalance
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < _MIN_HISTORY:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Get histories for both ETFs
        try:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
        except KeyError:
            return []

        if len(jnk_hist) < self.slow_ma + 1 or len(lqd_hist) < self.slow_ma + 1:
            return []

        # Compute ratio series: JNK price / LQD price over the lookback window
        jnk_close = jnk_hist["close"].iloc[-(self.slow_ma + 5):]
        lqd_close = lqd_hist["close"].iloc[-(self.slow_ma + 5):]

        min_len = min(len(jnk_close), len(lqd_close))
        if min_len < self.slow_ma:
            return []

        jnk_c = jnk_close.iloc[-min_len:].values
        lqd_c = lqd_close.iloc[-min_len:].values

        # Compute ratio
        ratio = jnk_c / lqd_c

        # Compute fast and slow MAs
        fast_ma_val = float(np.mean(ratio[-self.fast_ma:]))
        slow_ma_val = float(np.mean(ratio[-self.slow_ma:]))

        # Determine target: JNK when fast > slow (spreads tightening), LQD otherwise
        target_sym = "JNK" if fast_ma_val > slow_ma_val else "LQD"
        other_sym = "LQD" if target_sym == "JNK" else "JNK"

        closes = ctx.closes()
        if target_sym not in closes or closes[target_sym] <= 0:
            return []

        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)

        orders: list[Order] = []

        # Exit the other position if held
        other_pos = ctx.position(other_sym)
        if other_pos.size > 0:
            orders.append(Order(side=OrderSide.SELL, size=other_pos.size, symbol=other_sym))

        # Size the target position
        target_pos = ctx.position(target_sym)
        target_price = closes[target_sym]
        target_shares = int(equity * self.exposure / target_price)

        delta = target_shares - int(target_pos.size)
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))
        elif delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=target_sym))

        return orders


NAME = "credit_spread_hyg_lqd"
HYPOTHESIS = (
    "Credit spread regime timing via JNK/LQD ratio MA crossover: hold JNK when "
    "20-day MA of JNK/LQD ratio > 60-day MA (tightening spreads, risk-on); rotate "
    "to LQD when 20d MA < 60d MA (widening spreads, defensive). Weekly rebalance check."
)

STRATEGY = CreditSpreadHygLqd(
    fast_ma=20,
    slow_ma=90,
    rebalance=5,
    exposure=0.97,
)
