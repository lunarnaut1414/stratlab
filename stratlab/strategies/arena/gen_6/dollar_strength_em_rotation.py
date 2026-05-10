"""Dollar-strength EM rotation strategy.

Hypothesis: The US dollar regime drives cross-asset returns systematically.
  - Falling dollar (UUP 20d MA < 60d MA): EEM 60% + QQQ 37% (EM + tech outperform weak USD)
  - Rising dollar (UUP 20d MA >= 60d MA): SPY 60% + TLT 37% (domestic + bonds, EM headwind)
  - Weekly rebalance (every 5 bars)

Rationale: A weaker dollar reduces the currency drag on EM investments and
improves USD earnings for multinationals. QQQ (tech-heavy) benefits from
cheaper funding conditions that typically accompany falling-dollar periods.
Conversely, a rising dollar signals tightening global liquidity and creates
EM headwinds — rotate to domestic SPY + defensive TLT.

This signal is orthogonal to VIX, SKEW, credit-spread, and yield-curve
signals already on the leaderboard. UUP tracks the USD Index (DXY) via
futures contracts and has data back to 2007.

Diversification:
  - Different from all gen_5 strategies (none use USD/FX as primary signal)
  - EEM exposure not present in any leaderboard strategy
  - Two-state rotation is structurally simpler than multi-bucket VIX/SKEW
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 20      # 20d MA of UUP
SLOW_MA = 60      # 60d MA of UUP
REBALANCE = 5     # weekly
EXPOSURE = 0.97
_UUP = "UUP"


class DollarStrengthEMRotation(Strategy):
    """USD-regime rotation: weak dollar -> EEM+QQQ; strong dollar -> SPY+TLT."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Read UUP (USD ETF) price history to determine dollar trend
        try:
            uup_hist = ctx.history(_UUP)
        except KeyError:
            return []
        if uup_hist is None or len(uup_hist) < self.slow_ma + 2:
            return []

        uup_close = uup_hist["close"].dropna()
        if len(uup_close) < self.slow_ma:
            return []

        fast_val = float(uup_close.iloc[-self.fast_ma:].mean())
        slow_val = float(uup_close.iloc[-self.slow_ma:].mean())
        dollar_rising = fast_val >= slow_val  # USD strengthening = risk-off for EM

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target allocation
        if dollar_rising:
            # Rising USD: domestic equity + bonds
            allocations = [("SPY", 0.60 * self.exposure), ("TLT", 0.40 * self.exposure)]
        else:
            # Falling USD: EM + growth
            allocations = [("EEM", 0.60 * self.exposure), ("QQQ", 0.40 * self.exposure)]

        target: dict[str, float] = {}
        for sym, weight in allocations:
            if sym in closes_now.index:
                target[sym] = weight

        # If any target is missing, fallback to SPY
        if not target:
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "dollar_strength_em_rotation"
HYPOTHESIS = (
    "Dollar-strength EM rotation: use UUP 20d vs 60d MA crossover as USD regime; "
    "falling dollar hold EEM 60%+QQQ 37%; rising dollar hold SPY 60%+TLT 37%; "
    "captures USD-sensitive cross-asset rotation orthogonal to VIX and credit signals; "
    "weekly rebalance"
)

UNIVERSE = ["UUP", "EEM", "QQQ", "SPY", "TLT"]

STRATEGY = DollarStrengthEMRotation()
