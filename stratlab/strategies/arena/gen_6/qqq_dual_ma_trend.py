"""QQQ tech sector trend-following with dual MA confirmation.

Hypothesis: Use a two-level trend filter on QQQ to modulate between full-risk
QQQ allocation, neutral SPY, and defensive IEF.
  - QQQ strong uptrend (50d MA > 150d MA AND price > 50d MA): hold QQQ 97%
  - QQQ weakening (above 150d MA but below 50d MA): hold SPY 80%
  - QQQ downtrend (below 150d MA): hold IEF 97% (defensive)
  Rebalance every 5 bars (weekly), check state on every bar.

Rationale:
  QQQ captures the tech/growth premium that dominated 2010-2018. A dual MA
  confirmation (50/150 cross AND price-above-SMA) reduces false signals
  versus a single crossover. The 3-state exposure model (QQQ / SPY / IEF)
  generates a nuanced position that transitions smoothly between risk levels
  instead of binary on/off switches.

Diversification vs leaderboard:
  - QQQ as primary holding (not SPY or SP500 stocks) — different daily return.
  - 50/150 MA cross for tech trend signal vs SPY 200d SMA for market trend.
  - IEF as defensive (mid-duration) not TLT/SHY.
  - gen5_sma_cross_vix_gate uses 10d/30d SMA on SPY + VIX<25; this uses
    50d/150d on QQQ only (no VIX gate). Structurally different.
  - corr_to_top5 expected moderate-to-low since QQQ holdings include NVDA,
    AAPL etc. but NOT the broad SP500 cross-section.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 50       # QQQ 50d SMA
SLOW_MA = 150      # QQQ 150d SMA
REBALANCE_EVERY = 5  # weekly check
QQQ_EXPOSURE = 0.97
SPY_EXPOSURE = 0.80
IEF_EXPOSURE = 0.97


class QQQDualMATrend(Strategy):
    """QQQ 3-state trend: full QQQ / SPY / IEF based on 50/150 MA dual filter."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        rebalance_every: int = REBALANCE_EVERY,
        qqq_exposure: float = QQQ_EXPOSURE,
        spy_exposure: float = SPY_EXPOSURE,
        ief_exposure: float = IEF_EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            rebalance_every=rebalance_every,
            qqq_exposure=qqq_exposure,
            spy_exposure=spy_exposure,
            ief_exposure=ief_exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.rebalance_every = int(rebalance_every)
        self.qqq_exposure = float(qqq_exposure)
        self.spy_exposure = float(spy_exposure)
        self.ief_exposure = float(ief_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Compute QQQ 50d and 150d SMA
        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []
        if len(qqq_hist) < self.slow_ma + 2:
            return []

        qqq_close = qqq_hist["close"].dropna()
        if len(qqq_close) < self.slow_ma:
            return []

        qqq_price = float(qqq_close.iloc[-1])
        fast_val = float(qqq_close.iloc[-self.fast_ma:].mean())
        slow_val = float(qqq_close.iloc[-self.slow_ma:].mean())

        # Determine regime
        golden_cross = fast_val > slow_val         # 50d > 150d
        price_above_fast = qqq_price > fast_val    # price > 50d SMA
        above_slow = qqq_price > slow_val          # price > 150d SMA

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if golden_cross and price_above_fast:
            # Strong uptrend: QQQ full exposure
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.qqq_exposure
        elif above_slow:
            # Weakening but still above slow SMA: SPY at reduced exposure
            if "SPY" in closes_now.index:
                target["SPY"] = self.spy_exposure
        else:
            # Downtrend: defensive IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.ief_exposure

        # Build orders
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


NAME = "qqq_dual_ma_trend"
HYPOTHESIS = (
    "QQQ tech sector trend-following with dual MA confirmation: hold QQQ at 97% when "
    "QQQ 50d MA > 150d MA AND QQQ price > 50d MA (strong uptrend); hold SPY at 80% when "
    "QQQ above 150d MA but below 50d MA (weakening trend); hold IEF at 97% when QQQ "
    "below 150d MA; weekly rebalance; QQQ/IEF pairing not present in existing leaderboard."
)

UNIVERSE = ["QQQ", "SPY", "IEF"]

STRATEGY = QQQDualMATrend()
