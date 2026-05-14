"""XLY/XLP consumer sentiment ratio as growth barometer.

Hypothesis: The relative performance of consumer discretionary (XLY) vs
consumer staples (XLP) reflects risk appetite and economic growth expectations.
When cyclical consumers outperform defensive consumers, it signals risk-on
conditions; when staples lead, it signals defensive positioning.

Signal tiers:
  - XLY 20d return > XLP 20d return by >1.5% (strong risk-on): QQQ 97%
  - XLY 20d return > XLP 20d return by 0% to 1.5% (mild risk-on): SPY 97%
  - XLP 20d return > XLY 20d return (defensive demand): TLT 97%
  - Override: SPY below 150d SMA -> TLT 97% regardless

Rationale: XLY/XLP ratio captures the growth vs defensiveness of consumer
spending, which leads the broader economy. This is fundamentally different
from VIX (fear), credit spreads (financial stress), or breadth (participation)
signals. Consumer spending drives 70% of US GDP — where consumers allocate
cyclical vs defensive spending is a leading indicator.

Distinct from:
  - All VIX-based strategies (different signal domain)
  - All credit-spread (JNK/LQD) strategies
  - All breadth (RSP/IWM) strategies
  - gen5_tech_vs_defensive_rotation (uses XLK vs XLU, not XLY vs XLP)
  - gen6 strategies using DBC or yield curve

Weekly rebalance to capture signal changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
MOMENTUM_WINDOW = 20       # 20-day return comparison
TREND_WINDOW = 150         # SPY bear market gate
STRONG_RISKEON_THRESHOLD = 0.015  # >1.5% spread -> QQQ
EXPOSURE = 0.97

_SPY = "SPY"
_XLY = "XLY"
_XLP = "XLP"


class XLYXLPConsumerRegime(Strategy):
    """XLY/XLP consumer-cyclical-vs-staples regime: QQQ/SPY/TLT tri-state allocator.

    Uses 20d relative return between consumer discretionary and staples as a
    growth/defensive barometer to allocate to QQQ (strong risk-on), SPY (mild
    risk-on), or TLT (defensive / bear market).
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        strong_riskeon_threshold: float = STRONG_RISKEON_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            strong_riskeon_threshold=strong_riskeon_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.strong_riskeon_threshold = float(strong_riskeon_threshold)
        self.exposure = float(exposure)

    def _compute_return(self, hist: pd.DataFrame, window: int) -> float | None:
        """Compute the N-day return from history frame. Returns None if insufficient data."""
        if hist is None or len(hist) < window + 1:
            return None
        close = hist["close"].dropna()
        if len(close) < window + 1:
            return None
        p_end = float(close.iloc[-1])
        p_start = float(close.iloc[-(window + 1)])
        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
            return None
        return (p_end / p_start) - 1.0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY bear market gate (150d SMA)
        spy_bull = False
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # XLY and XLP 20d returns
        xly_ret = None
        xlp_ret = None
        try:
            xly_hist = ctx.history(_XLY)
            xly_ret = self._compute_return(xly_hist, self.momentum_window)
        except Exception:
            pass
        try:
            xlp_hist = ctx.history(_XLP)
            xlp_ret = self._compute_return(xlp_hist, self.momentum_window)
        except Exception:
            pass

        # Determine allocation target
        target: dict[str, float] = {}

        if not spy_bull or xly_ret is None or xlp_ret is None:
            # Bear market or missing data — defensive
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            spread = xly_ret - xlp_ret
            if spread > self.strong_riskeon_threshold:
                # Strong risk-on: QQQ
                if "QQQ" in live:
                    target["QQQ"] = self.exposure
                elif "SPY" in live:
                    target["SPY"] = self.exposure
            elif spread > 0:
                # Mild risk-on: SPY
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                # Defensive demand (XLP leading): TLT
                if "TLT" in live:
                    target["TLT"] = self.exposure

        # Build orders
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


NAME = "xly_xlp_consumer_regime"
HYPOTHESIS = (
    "XLY/XLP consumer sentiment ratio as growth barometer: hold QQQ 97% when XLY 20d return "
    "> XLP 20d return by >1.5% (risk appetite), hold SPY 97% when XLY modestly leads, TLT "
    "97% when XLP leads (defensive demand) or SPY below 150d SMA; weekly rebalance; "
    "consumer-cyclicals-vs-staples regime orthogonal to credit/VIX/breadth signals"
)

UNIVERSE = ["SPY", "QQQ", "TLT", "XLY", "XLP"]

STRATEGY = XLYXLPConsumerRegime()
