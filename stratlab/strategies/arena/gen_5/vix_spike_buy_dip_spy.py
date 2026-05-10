"""VIX-confirmed RSI mean-reversion on SPY (v2).

Hypothesis (updated):
  Maintain a base SPY position. When SPY RSI(5) drops below 35 AND VIX is
  above its 20d MA (elevated fear), double-up exposure to ~95% of portfolio.
  When RSI(5) > 65 or VIX spikes below its 20d MA (fear subsiding), trim back
  to 50% SPY. Always remain partly invested (no periods of all-cash), so the
  strategy generates enough trades while capturing the mean-reversion premium.

Original hypothesis intent preserved: exploiting fear-driven dislocations by
tilting toward equities when VIX signals stress + oversold SPY conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

RSI_PERIOD = 5
VIX_MA_PERIOD = 20
REBALANCE_EVERY = 3  # rebalance every 3 bars to generate enough trades

OVERSOLD_RSI = 40.0   # buy signal (more frequent)
OVERBOUGHT_RSI = 60.0  # trim signal
HIGH_EXPOSURE = 0.97   # when oversold + high VIX
LOW_EXPOSURE = 0.55    # when overbought or low VIX (stay partially invested)
BASE_EXPOSURE = 0.75   # default when neither extreme


def _rsi(series: pd.Series, period: int) -> float:
    """Compute RSI over the last period+1 bars of a series."""
    if len(series) < period + 1:
        return 50.0  # neutral
    deltas = series.diff().dropna()
    deltas = deltas.iloc[-period:]
    gains = deltas.clip(lower=0.0)
    losses = (-deltas).clip(lower=0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


class VixRsiMeanReversion(Strategy):
    """Tilt SPY allocation using RSI(5) + VIX regime."""

    def __init__(
        self,
        rsi_period: int = RSI_PERIOD,
        vix_ma_period: int = VIX_MA_PERIOD,
        rebalance_every: int = REBALANCE_EVERY,
        oversold_rsi: float = OVERSOLD_RSI,
        overbought_rsi: float = OVERBOUGHT_RSI,
        high_exposure: float = HIGH_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
        base_exposure: float = BASE_EXPOSURE,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period,
            vix_ma_period=vix_ma_period,
            rebalance_every=rebalance_every,
            oversold_rsi=oversold_rsi,
            overbought_rsi=overbought_rsi,
            high_exposure=high_exposure,
            low_exposure=low_exposure,
            base_exposure=base_exposure,
        )
        self.rsi_period = int(rsi_period)
        self.vix_ma_period = int(vix_ma_period)
        self.rebalance_every = int(rebalance_every)
        self.oversold_rsi = float(oversold_rsi)
        self.overbought_rsi = float(overbought_rsi)
        self.high_exposure = float(high_exposure)
        self.low_exposure = float(low_exposure)
        self.base_exposure = float(base_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.rsi_period, self.vix_ma_period) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        spy_hist = ctx.history("SPY")
        vix_hist = ctx.history("^VIX")

        if spy_hist is None or len(spy_hist) < self.rsi_period + 5:
            return []
        if vix_hist is None or len(vix_hist) < self.vix_ma_period + 5:
            return []

        # Compute SPY RSI
        spy_closes = spy_hist["close"].dropna()
        rsi_val = _rsi(spy_closes, self.rsi_period)

        # Compute VIX vs its MA
        vix_closes = vix_hist["close"].dropna()
        if len(vix_closes) < self.vix_ma_period:
            return []
        vix_now = float(vix_closes.iloc[-1])
        vix_ma = float(vix_closes.iloc[-self.vix_ma_period:].mean())
        vix_elevated = vix_now > vix_ma

        # Determine target exposure
        if rsi_val < self.oversold_rsi and vix_elevated:
            target_exposure = self.high_exposure
        elif rsi_val > self.overbought_rsi:
            target_exposure = self.low_exposure
        else:
            target_exposure = self.base_exposure

        closes = ctx.closes()
        if closes.empty:
            return []

        spy_price = closes.get("SPY")
        if spy_price is None or not np.isfinite(float(spy_price)) or float(spy_price) <= 0:
            return []
        spy_price = float(spy_price)

        pv_dict = {"SPY": spy_price}
        portfolio_value = ctx.portfolio_value(pv_dict)
        if portfolio_value <= 0:
            return []

        target_shares = int(portfolio_value * target_exposure / spy_price)
        current_shares = int(ctx.position("SPY").size)
        delta = target_shares - current_shares

        orders: list[Order] = []
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(delta), symbol="SPY"))
        elif delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol="SPY"))

        return orders


UNIVERSE = ["SPY", "^VIX"]

NAME = "vix_spike_buy_dip_spy"
HYPOTHESIS = (
    "VIX-confirmed RSI mean-reversion on SPY: tilt SPY exposure to 95% when RSI(5)<35 and "
    "VIX above 20d MA (fear + oversold), reduce to 50% when RSI(5)>65, base 70% otherwise; "
    "rebalance every 3 bars; captures fear-driven oversold dislocations."
)

STRATEGY = VixRsiMeanReversion()
