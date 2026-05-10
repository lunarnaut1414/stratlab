"""Consumer Discretionary vs Staples Rotation with Trend Confirmation — gen_5 sonnet-9

Hypothesis:
  The XLY/VDC ratio (consumer discretionary / consumer staples) captures
  risk appetite shifts. Combined with an SPY absolute momentum filter,
  this avoids switching to bonds during minor corrections.

  Regime logic (biweekly rebalance):
  1. SPY above 200d SMA AND XLY outperforming VDC on 20d return
     -> hold QQQ (risk-on, high growth)
  2. SPY above 200d SMA but VDC outperforming XLY
     -> hold SPY (cautious equity, reducing beta slightly)
  3. SPY below 200d SMA (trend break confirmed)
     -> hold TLT (defensive bonds)

  The layered signal reduces false defensive pivots from mere sector rotation
  noise, and uses consumer sentiment as a fine-tuning mechanism within the
  primary equity trend framework.

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["XLY", "VDC", "SPY", "QQQ", "TLT"]

MOMENTUM_WINDOW = 20   # 20-day return comparison for XLY vs VDC
TREND_WINDOW = 200     # 200d SMA for SPY trend confirmation
REBALANCE_DAYS = 10    # biweekly rebalance
MIN_HISTORY = TREND_WINDOW + 10
EXPOSURE = 0.97


class XlyVdcConsumerRotation(Strategy):
    """Consumer sentiment + trend confirmation sector rotation."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_days: int = REBALANCE_DAYS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            trend_window=trend_window,
            rebalance_days=rebalance_days,
            exposure=exposure,
        )
        self.momentum_window = momentum_window
        self.trend_window = trend_window
        self.rebalance_days = rebalance_days
        self.exposure = exposure
        self._bar_count: int = 0

    def on_start(self) -> None:
        self._bar_count = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < MIN_HISTORY:
            return []

        self._bar_count += 1
        if self._bar_count % self.rebalance_days != 0:
            return []

        available = set(ctx.symbols)

        # Need SPY for trend signal
        if "SPY" not in available:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []

        if len(spy_hist) < self.trend_window + 1:
            return []

        spy_close = spy_hist["close"]
        spy_current = float(spy_close.iloc[-1])
        spy_200ma = float(spy_close.iloc[-self.trend_window:].mean())

        if not np.isfinite(spy_current) or not np.isfinite(spy_200ma):
            return []

        spy_above_trend = spy_current > spy_200ma

        # Consumer sentiment signal (optional but strengthens regime)
        xly_ret, vdc_ret = None, None
        if "XLY" in available and "VDC" in available:
            try:
                xly_hist = ctx.history("XLY")
                vdc_hist = ctx.history("VDC")
                if len(xly_hist) >= self.momentum_window + 1 and len(vdc_hist) >= self.momentum_window + 1:
                    xly_ret = float(xly_hist["close"].iloc[-1] / xly_hist["close"].iloc[-self.momentum_window] - 1.0)
                    vdc_ret = float(vdc_hist["close"].iloc[-1] / vdc_hist["close"].iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(xly_ret) or not np.isfinite(vdc_ret):
                        xly_ret, vdc_ret = None, None
            except KeyError:
                pass

        closes = ctx.closes()
        if closes.empty:
            return []

        live_closes = {s: float(closes[s]) for s in closes.index if closes[s] > 0}
        equity = ctx.portfolio_value(live_closes)
        if equity <= 0:
            return []

        # Three-state regime:
        if not spy_above_trend:
            # Primary bear signal: SPY below 200d SMA -> defensive
            target_sym = "TLT"
        elif xly_ret is not None and vdc_ret is not None and xly_ret > vdc_ret:
            # SPY bullish + consumer discretionary leading: aggressive
            target_sym = "QQQ"
        else:
            # SPY bullish but consumer cautious or no signal: moderate equity
            target_sym = "SPY"

        # Fallback
        if target_sym not in available or live_closes.get(target_sym, 0) <= 0:
            for fallback in ["SPY", "QQQ", "TLT"]:
                if fallback in available and live_closes.get(fallback, 0) > 0:
                    target_sym = fallback
                    break
            else:
                return []

        target_price = live_closes[target_sym]
        target_shares = int(equity * self.exposure / target_price)
        if target_shares < 1:
            return []

        orders: list[Order] = []

        # Exit all other positions
        for sym, pos in list(ctx.positions.items()):
            if sym != target_sym and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Size the target
        current = int(ctx.position(target_sym).size)
        delta = target_shares - current
        if delta != 0:
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=target_sym))

        return orders


NAME = "xly_xlp_consumer_rotation"
HYPOTHESIS = (
    "Consumer discretionary vs staples (XLY/VDC) with SPY 200d trend confirmation: "
    "hold QQQ when SPY>200d and XLY outperforms VDC; hold SPY when bullish trend but "
    "consumer cautious; hold TLT when SPY below 200d SMA; biweekly rebalance."
)

STRATEGY = XlyVdcConsumerRotation()
