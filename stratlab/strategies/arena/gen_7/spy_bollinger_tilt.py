"""SPY Bollinger Band Tiered Exposure with QQQ/IWM tilt — gen_7 sonnet-3

Hypothesis: Use SPY Bollinger Band pct-b as a tiered exposure signal.
When SPY pct-b < 0.2 (oversold) AND SPY above 200d SMA, tilt to
QQQ 80%+IWM 17% (risk-on beta chase). When pct-b > 0.8 (overbought),
reduce to SPY 60% (take some risk off). Else SPY 95%. Weekly rebalance.

Rationale: Bollinger pct-b captures relative position within the recent
volatility range. Oversold within an uptrend suggests a bounce setup;
overbought within the same trend suggests a temporary profit-taking.
Routing to QQQ+IWM on dips maximizes the bounce participation.
TLT defensive when SPY below 200d SMA.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
BB_PERIOD = 20            # Bollinger band lookback
BB_STD = 2.0              # Bollinger band std multiplier
TREND_WINDOW = 200        # 200d SMA for SPY
OVERSOLD_THRESHOLD = 0.2  # pct-b below this -> risk-on boost
OVERBOUGHT_THRESHOLD = 0.8  # pct-b above this -> reduce exposure
HIGH_EXPOSURE = 0.97
BASE_EXPOSURE = 0.95
LOW_EXPOSURE = 0.60
QQQ_WEIGHT = 0.78
IWM_WEIGHT = 0.19


class SpyBollingerTilt(Strategy):
    """SPY Bollinger Band tiered QQQ/IWM tilt strategy."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
        trend_window: int = TREND_WINDOW,
        oversold_threshold: float = OVERSOLD_THRESHOLD,
        overbought_threshold: float = OVERBOUGHT_THRESHOLD,
        high_exposure: float = HIGH_EXPOSURE,
        base_exposure: float = BASE_EXPOSURE,
        low_exposure: float = LOW_EXPOSURE,
        qqq_weight: float = QQQ_WEIGHT,
        iwm_weight: float = IWM_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            bb_period=bb_period,
            bb_std=bb_std,
            trend_window=trend_window,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            high_exposure=high_exposure,
            base_exposure=base_exposure,
            low_exposure=low_exposure,
            qqq_weight=qqq_weight,
            iwm_weight=iwm_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.trend_window = int(trend_window)
        self.oversold_threshold = float(oversold_threshold)
        self.overbought_threshold = float(overbought_threshold)
        self.high_exposure = float(high_exposure)
        self.base_exposure = float(base_exposure)
        self.low_exposure = float(low_exposure)
        self.qqq_weight = float(qqq_weight)
        self.iwm_weight = float(iwm_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.bb_period + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY history for trend and Bollinger
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + self.bb_period + 5:
            return []

        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []

        # SPY 200d SMA trend gate
        spy_sma200 = float(spy_close.iloc[-self.trend_window:].mean())
        spy_last = float(spy_close.iloc[-1])
        bull = spy_last > spy_sma200

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market — defensive TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.high_exposure
        else:
            # Compute Bollinger Band pct-b on SPY
            if len(spy_close) < self.bb_period + 2:
                return []
            recent = spy_close.iloc[-self.bb_period:]
            bb_mean = float(recent.mean())
            bb_std = float(recent.std(ddof=1))

            if bb_std <= 1e-6 or not np.isfinite(bb_std):
                pct_b = 0.5
            else:
                upper = bb_mean + self.bb_std * bb_std
                lower = bb_mean - self.bb_std * bb_std
                band_range = upper - lower
                if band_range <= 0:
                    pct_b = 0.5
                else:
                    pct_b = (spy_last - lower) / band_range

            if pct_b < self.oversold_threshold:
                # Oversold within uptrend: risk-on tilt to QQQ+IWM
                if "QQQ" in closes_now.index and "IWM" in closes_now.index:
                    target["QQQ"] = self.high_exposure * self.qqq_weight
                    target["IWM"] = self.high_exposure * self.iwm_weight
                elif "QQQ" in closes_now.index:
                    target["QQQ"] = self.high_exposure
                else:
                    target["SPY"] = self.high_exposure
            elif pct_b > self.overbought_threshold:
                # Overbought: reduce risk, hold SPY at lower exposure
                if "SPY" in closes_now.index:
                    target["SPY"] = self.low_exposure
            else:
                # Neutral: hold SPY at base exposure
                if "SPY" in closes_now.index:
                    target["SPY"] = self.base_exposure

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


UNIVERSE = ["SPY", "QQQ", "IWM", "TLT"]

NAME = "spy_bollinger_tilt"
HYPOTHESIS = (
    "Bollinger Band mean-reversion on popular ETFs: when SPY BB pct-b < 0.2 (oversold within trend) "
    "AND SPY above 200d SMA, tilt to QQQ 80%+IWM 17%; when BB pct-b > 0.8 (overbought) reduce to "
    "SPY 60%; else SPY 95%; weekly rebalance; Bollinger pct-b as continuous exposure scalar on equity allocation"
)

STRATEGY = SpyBollingerTilt()
