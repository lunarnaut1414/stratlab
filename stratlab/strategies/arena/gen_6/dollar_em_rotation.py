"""Dollar-strength EM rotation strategy.

Hypothesis: Use UUP (dollar ETF) 20d vs 60d MA crossover as the dollar regime
signal. When the dollar is falling (UUP < 60d MA), hold EEM 60% + QQQ 37%.
When dollar is rising (UUP > 60d MA), hold SPY 60% + TLT 37%.

Rationale: A weak dollar is a tailwind for emerging markets (EM) and
risk assets denominated in foreign currencies. A strong dollar tends to
drag EM and boost defensive dollar assets. This cross-asset rotation is
orthogonal to VIX/credit spread signals on the existing leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["UUP", "EEM", "QQQ", "SPY", "TLT"]

FAST_MA = 20
SLOW_MA = 60
REBALANCE_EVERY = 5  # weekly rebalance
EXPOSURE = 0.97


class DollarEmRotation(Strategy):
    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get UUP history for dollar signal
        try:
            uup_hist = ctx.history("UUP")
        except KeyError:
            return []
        if uup_hist is None or len(uup_hist) < self.slow_ma + 2:
            return []

        uup_close = uup_hist["close"].dropna()
        if len(uup_close) < self.slow_ma:
            return []

        uup_now = float(uup_close.iloc[-1])
        uup_slow_ma = float(uup_close.iloc[-self.slow_ma:].mean())

        # Dollar regime: rising = UUP above slow MA, falling = below
        dollar_rising = uup_now > uup_slow_ma

        # Define target weights based on dollar regime
        if dollar_rising:
            # Strong dollar: favor domestic, add bonds
            target_weights = {
                "SPY": 0.60 * self.exposure,
                "TLT": 0.40 * self.exposure,
            }
        else:
            # Weak dollar: favor EM and Nasdaq growth
            target_weights = {
                "EEM": 0.60 * self.exposure,
                "QQQ": 0.40 * self.exposure,
            }

        # Get current prices
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_weights and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target_weights.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "dollar_em_rotation"
HYPOTHESIS = (
    "Dollar-strength EM rotation: use UUP 20d vs 60d MA crossover as dollar "
    "regime signal; when dollar falling (UUP<60d MA) hold EEM 60%+QQQ 37%; "
    "when dollar rising hold SPY 60%+TLT 37%; captures USD-sensitive cross-asset "
    "rotation orthogonal to VIX/credit signals"
)

STRATEGY = DollarEmRotation()
