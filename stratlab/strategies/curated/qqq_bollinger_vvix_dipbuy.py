"""opus-1 mutation of vix_spike_buy_dip_spy (parent IS Calmar 0.55, corr 0.71).

Continuous QQQ dip-buy via Bollinger %b with VVIX (vol-of-vol) regime gate.

Structural mutations vs parent (SPY/RSI(5)/VIX-vs-20dMA/3-bar rebalance):
  - Asset:        SPY  ->  QQQ (Nasdaq-100; higher beta, different dip
                            statistics).
  - Oscillator:   RSI(5)  ->  Bollinger %b (price relative to 20d mean
                              ± 2 std). %b<0 = below lower band, %b>1
                              = above upper band. Different mean-reversion
                              statistic — uses standard deviation instead of
                              up/down rank.
  - Vol gate:     VIX vs its 20d MA  ->  ^VVIX vs its 90d MA. VVIX is
                                          vol-of-vol — fires on different
                                          days than VIX-level filters.
  - Rebalance:    3 bars  ->  5 bars (less churn).
  - Tilt levels:  base 75% / high 97% / low 55%  ->  base 70% / high 97%
                                                     / low 45% (more
                                                     aggressive trim).

The combination of %b (continuous) instead of RSI (rank-based), VVIX (vol-of-
vol) instead of VIX (vol level), and QQQ (high-beta) instead of SPY means
daily returns will differ structurally from the parent's path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "^VVIX"]

BB_PERIOD = 20
BB_STD = 2.0
VVIX_MA_PERIOD = 90
REBALANCE_EVERY = 5

LOWER_BAND_TRIGGER = 0.0   # %b < 0  -> oversold
UPPER_BAND_TRIGGER = 1.0   # %b > 1  -> overbought
HIGH_EXPOSURE = 0.97
LOW_EXPOSURE = 0.45
BASE_EXPOSURE = 0.70


def _bollinger_pctb(closes: pd.Series, period: int, num_std: float) -> float:
    if len(closes) < period:
        return 0.5
    tail = closes.iloc[-period:]
    mu = float(tail.mean())
    sd = float(tail.std(ddof=0))
    if sd <= 1e-9:
        return 0.5
    upper = mu + num_std * sd
    lower = mu - num_std * sd
    last = float(closes.iloc[-1])
    return (last - lower) / (upper - lower) if upper > lower else 0.5


class QqqBollingerVvixDipBuy(Strategy):
    def __init__(
        self,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
        vvix_ma_period: int = VVIX_MA_PERIOD,
        rebalance_every: int = REBALANCE_EVERY,
        lower_band_trigger: float = LOWER_BAND_TRIGGER,
        upper_band_trigger: float = UPPER_BAND_TRIGGER,
        high_exposure: float = HIGH_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
        base_exposure: float = BASE_EXPOSURE,
    ) -> None:
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            vvix_ma_period=vvix_ma_period,
            rebalance_every=rebalance_every,
            lower_band_trigger=lower_band_trigger,
            upper_band_trigger=upper_band_trigger,
            high_exposure=high_exposure,
            low_exposure=low_exposure,
            base_exposure=base_exposure,
        )
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.vvix_ma_period = int(vvix_ma_period)
        self.rebalance_every = int(rebalance_every)
        self.lower_band_trigger = float(lower_band_trigger)
        self.upper_band_trigger = float(upper_band_trigger)
        self.high_exposure = float(high_exposure)
        self.low_exposure = float(low_exposure)
        self.base_exposure = float(base_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.bb_period, self.vvix_ma_period) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []
        if qqq_hist is None or len(qqq_hist) < self.bb_period + 5:
            return []
        qclose = qqq_hist["close"].dropna()
        pctb = _bollinger_pctb(qclose, self.bb_period, self.bb_std)

        # VVIX regime
        vvix_elevated = False
        try:
            vvix_hist = ctx.history("^VVIX")
            if vvix_hist is not None and len(vvix_hist) >= self.vvix_ma_period + 1:
                vc = vvix_hist["close"].dropna()
                vnow = float(vc.iloc[-1])
                vma = float(vc.iloc[-self.vvix_ma_period:].mean())
                if np.isfinite(vnow) and np.isfinite(vma):
                    vvix_elevated = vnow > vma
        except KeyError:
            pass

        # Determine target exposure
        if pctb < self.lower_band_trigger and vvix_elevated:
            target_exposure = self.high_exposure
        elif pctb > self.upper_band_trigger:
            target_exposure = self.low_exposure
        else:
            target_exposure = self.base_exposure

        live = ctx.closes()
        qprice = live.get("QQQ")
        if qprice is None or not np.isfinite(float(qprice)) or float(qprice) <= 0:
            return []
        qprice = float(qprice)
        equity = ctx.portfolio_value({"QQQ": qprice})
        if equity <= 0:
            return []

        target_shares = int(equity * target_exposure / qprice)
        cur_shares = int(ctx.position("QQQ").size)
        delta = target_shares - cur_shares
        if abs(delta) < 1:
            return []

        orders: list[Order] = []
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(delta), symbol="QQQ"))
        else:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol="QQQ"))
        return orders


NAME = "opus1_qqq_bollinger_vvix_dipbuy"
HYPOTHESIS = (
    "Mutate vix_spike_buy_dip_spy to continuous QQQ dip-buy via Bollinger %b: "
    "tilt 97% when %b<0 and VVIX>90d MA, base 70%, trim to 45% when %b>1; "
    "rebalance every 5 bars; VVIX (vol-of-vol) gating instead of VIX level."
)

STRATEGY = QqqBollingerVvixDipBuy()
