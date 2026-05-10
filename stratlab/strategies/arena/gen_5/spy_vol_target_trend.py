"""SPY volatility-targeted 200d trend strategy.

Hypothesis:
  - Hold SPY when SPY close > 200-day SMA (bull regime).
  - Position sized to target ~10% annualized vol (using 20-day realized vol).
  - When SPY < 200d SMA (bear regime), hold SHY (short-term Treasuries).
  - Daily rebalance to maintain vol target.

Rationale: The 200d SMA is one of the most robust trend signals in equity
markets. Overlaying volatility targeting (scale down when vol is high, scale
up when low) reduces drawdowns without sacrificing much upside. This is
distinct from buy-and-hold and from any momentum cross-section strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "SHY"]

_SPY = "SPY"
_SHY = "SHY"
_TREND_WINDOW = 200
_VOL_WINDOW = 20
_TARGET_VOL = 0.10          # 10% annualized target vol
_MAX_EXPOSURE = 1.5         # cap leverage at 1.5x
_MIN_EXPOSURE = 0.10        # always keep at least 10% in SPY during bull
_REBALANCE_THRESHOLD = 0.05 # only rebalance if weight drift > 5%


class SpyVolTargetTrend(Strategy):
    """SPY 200d trend filter + vol-targeting position sizer."""

    def __init__(
        self,
        trend_window: int = _TREND_WINDOW,
        vol_window: int = _VOL_WINDOW,
        target_vol: float = _TARGET_VOL,
        max_exposure: float = _MAX_EXPOSURE,
        min_exposure: float = _MIN_EXPOSURE,
        rebalance_threshold: float = _REBALANCE_THRESHOLD,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            vol_window=vol_window,
            target_vol=target_vol,
            max_exposure=max_exposure,
            min_exposure=min_exposure,
            rebalance_threshold=rebalance_threshold,
        )
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_exposure = max_exposure
        self.min_exposure = min_exposure
        self.rebalance_threshold = rebalance_threshold

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Need enough history for trend window
        if ctx.idx < self.trend_window + self.vol_window + 2:
            return []

        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.trend_window + self.vol_window:
            return []

        spy_closes = spy_hist["close"]
        spy_close = float(spy_closes.iloc[-1])
        spy_sma200 = float(spy_closes.iloc[-self.trend_window:].mean())
        bull_regime = spy_close > spy_sma200

        live_closes = ctx.closes()
        live_dict = {s: float(p) for s, p in live_closes.items()}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []
        target: dict[str, int] = {}

        if not bull_regime:
            # Bear: move to SHY
            shy_price = live_dict.get(_SHY, 0.0)
            if shy_price > 0:
                shares = int(equity * 0.98 / shy_price)
                if shares > 0:
                    target[_SHY] = shares
        else:
            # Bull: vol-targeted SPY position
            log_rets = np.log(
                spy_closes.iloc[-self.vol_window - 1:].values[1:] /
                spy_closes.iloc[-self.vol_window - 1:].values[:-1]
            )
            realized_vol = float(np.std(log_rets)) * np.sqrt(252)
            if realized_vol < 1e-6:
                return []

            # Scale exposure: target_vol / realized_vol, clamped
            raw_exposure = self.target_vol / realized_vol
            exposure = min(self.max_exposure, max(self.min_exposure, raw_exposure))

            spy_price = live_dict.get(_SPY, 0.0)
            if spy_price > 0:
                target_notional = equity * exposure
                shares = int(target_notional / spy_price)
                if shares > 0:
                    target[_SPY] = shares

        # Compute current SPY/SHY weight vs target to decide if rebalance needed
        for sym, tgt_shares in target.items():
            cur_shares = ctx.position(sym).size
            price = live_dict.get(sym, 0.0)
            if price > 0 and equity > 0:
                tgt_weight = tgt_shares * price / equity
                cur_weight = cur_shares * price / equity
                if abs(tgt_weight - cur_weight) < self.rebalance_threshold:
                    target[sym] = int(cur_shares)  # no-op: keep existing

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        # Adjust to target
        for sym, tgt in target.items():
            current = int(ctx.position(sym).size)
            delta = tgt - current
            if delta == 0:
                continue
            if delta > 0:
                orders.append(Order(side=OrderSide.BUY, size=float(delta), symbol=sym))
            else:
                orders.append(Order(side=OrderSide.SELL, size=float(-delta), symbol=sym))

        return orders


NAME = "spy_vol_target_trend"
HYPOTHESIS = (
    "SPY 200d SMA trend filter with volatility targeting: hold SPY when above "
    "200d SMA sized to 10% annualized vol target; hold SHY when below; adjusts "
    "position daily based on 20-day realized vol."
)

STRATEGY = SpyVolTargetTrend()
