"""QQQ RSI trend-dip momentum — gen_6 sonnet-7

Hypothesis: Hold QQQ with exposure adjusted by RSI(10) oscillator AND QQQ
200d SMA trend filter:
  - QQQ above 200d SMA (bull trend):
      * RSI(10) < 35 (oversold dip in uptrend): increase to 97% → buy the dip
      * RSI(10) 35-65 (neutral): hold at 85%
      * RSI(10) > 65 (overbought): reduce to 60%
  - QQQ below 200d SMA (bear trend): hold TLT 97%
  Rebalance every 3 bars (short cycle to catch dips quickly).

Rationale:
  QQQ in 2010-2018 had a strong upward trend punctuated by brief oversold
  dips. Buying dips in an uptrend (RSI<35 + price>200d SMA) captures the
  mean-reversion premium while staying long in the dominant trend. Trimming
  when overbought reduces drawdown risk. Fully different from VIX/VVIX-gated
  strategies: uses RSI directly on QQQ (not SPY).

  Distinct from existing leaderboard:
  - Single-asset QQQ with RSI oscillator (no sector or stock selection)
  - Trend + RSI combination (not VIX gating)
  - QQQ 200d SMA (not SPY 200d SMA as gate)
  - 3-bar rebalance cycle for responsive dip-buying
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 3
RSI_PERIOD = 10
TREND_WINDOW = 200
OVERSOLD_RSI = 35.0
OVERBOUGHT_RSI = 65.0
HIGH_EXPOSURE = 0.97
MID_EXPOSURE = 0.85
LOW_EXPOSURE = 0.60
EXPOSURE = 0.97


def _compute_rsi(closes: "np.ndarray", period: int) -> float:
    """Compute RSI from closing price array."""
    if len(closes) < period + 1:
        return 50.0  # neutral default
    diffs = np.diff(closes[-(period + 1):])
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


class QQQRSITrendDip(Strategy):
    """QQQ trend-following with RSI-based exposure tilt."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rsi_period: int = RSI_PERIOD,
        trend_window: int = TREND_WINDOW,
        oversold_rsi: float = OVERSOLD_RSI,
        overbought_rsi: float = OVERBOUGHT_RSI,
        high_exposure: float = HIGH_EXPOSURE,
        mid_exposure: float = MID_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rsi_period=rsi_period,
            trend_window=trend_window,
            oversold_rsi=oversold_rsi,
            overbought_rsi=overbought_rsi,
            high_exposure=high_exposure,
            mid_exposure=mid_exposure,
            low_exposure=low_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.rsi_period = int(rsi_period)
        self.trend_window = int(trend_window)
        self.oversold_rsi = float(oversold_rsi)
        self.overbought_rsi = float(overbought_rsi)
        self.high_exposure = float(high_exposure)
        self.mid_exposure = float(mid_exposure)
        self.low_exposure = float(low_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.rsi_period + 5
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

        # --- QQQ trend filter (200d SMA) ---
        qqq_bull = False
        rsi_val = 50.0
        try:
            qqq_hist = ctx.history("QQQ")
            if len(qqq_hist) >= self.trend_window + self.rsi_period + 5:
                qqq_close = qqq_hist["close"].dropna().values
                sma = float(np.mean(qqq_close[-self.trend_window:]))
                qqq_bull = float(qqq_close[-1]) > sma
                rsi_val = _compute_rsi(qqq_close, self.rsi_period)
        except Exception:
            return []

        target: dict[str, float] = {}

        if not qqq_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = 0.97
        else:
            # Bull market: QQQ with RSI-tiered exposure
            if rsi_val < self.oversold_rsi:
                exp = self.high_exposure   # dip in uptrend
            elif rsi_val > self.overbought_rsi:
                exp = self.low_exposure    # overbought, trim
            else:
                exp = self.mid_exposure    # neutral

            if "QQQ" in closes_now.index:
                target["QQQ"] = exp

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


NAME = "qqq_rsi_trend_dip"
HYPOTHESIS = (
    "QQQ RSI trend-dip: hold QQQ 97% when QQQ>200d SMA and RSI(10)<35 (oversold dip); "
    "85% when neutral RSI 35-65; 60% when RSI>65 (overbought); TLT 97% when QQQ<200d SMA; "
    "3-bar rebalance; QQQ 200d SMA + RSI tilt, not VIX-gated"
)
UNIVERSE = ["QQQ", "TLT"]
STRATEGY = QQQRSITrendDip()
