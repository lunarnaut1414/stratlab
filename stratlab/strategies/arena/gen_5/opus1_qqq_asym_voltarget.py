"""opus-1 mutation of spy_vol_target_trend (parent IS Calmar 0.52, corr 0.47).

Structural mutations:
  - Asset:        SPY  ->  SPY (kept).
  - Trend filter: SPY 200d SMA  ->  SPY 50d SMA crossover with SPY 150d SMA
                                     (golden-cross style trend signal — fast
                                     vs medium MA cross instead of price vs
                                     long MA).
  - Vol horizon:  20d realized vol  ->  60d realized vol.
  - Target vol:   10%  ->  11%.
  - Max leverage: 1.5x  ->  1.5x (kept; the structural change is in the
                                   trend signal, not the sizing cap).
  - Defensive:    SHY  ->  TLT (long-duration treasury — picks up duration
                                premium which can offset equity drawdowns).
  - Min exposure when bull: 10%  ->  20%.
  - Rebalance threshold: 5% drift  ->  5% drift (kept).

Cross-MA trend signal + 60d-vol horizon + TLT defensive bucket gives a
distinctly different daily return path than the parent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["RSP", "AGG"]

_EQUITY = "RSP"
_BOND = "AGG"
_FAST_MA = 50
_SLOW_MA = 150
_VOL_WINDOW = 60
_TARGET_VOL = 0.11
_MAX_EXPOSURE = 1.5
_MIN_EXPOSURE = 0.20
_REBALANCE_THRESHOLD = 0.05


class SpyCrossMAVolTarget(Strategy):
    def __init__(
        self,
        fast_ma: int = _FAST_MA,
        slow_ma: int = _SLOW_MA,
        vol_window: int = _VOL_WINDOW,
        target_vol: float = _TARGET_VOL,
        max_exposure: float = _MAX_EXPOSURE,
        min_exposure: float = _MIN_EXPOSURE,
        rebalance_threshold: float = _REBALANCE_THRESHOLD,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            vol_window=vol_window,
            target_vol=target_vol,
            max_exposure=max_exposure,
            min_exposure=min_exposure,
            rebalance_threshold=rebalance_threshold,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.vol_window = int(vol_window)
        self.target_vol = float(target_vol)
        self.max_exposure = float(max_exposure)
        self.min_exposure = float(min_exposure)
        self.rebalance_threshold = float(rebalance_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < self.slow_ma + self.vol_window + 2:
            return []

        try:
            eq_hist = ctx.history(_EQUITY)
        except KeyError:
            return []
        if len(eq_hist) < self.slow_ma + self.vol_window:
            return []

        eclose = eq_hist["close"].dropna()
        fast = float(eclose.iloc[-self.fast_ma:].mean())
        slow = float(eclose.iloc[-self.slow_ma:].mean())
        bull = fast > slow

        live = ctx.closes()
        live_dict = {s: float(p) for s, p in live.items()}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        target: dict[str, int] = {}
        if not bull:
            bond_p = live_dict.get(_BOND, 0.0)
            if bond_p > 0:
                target[_BOND] = int(equity * 0.97 / bond_p)
        else:
            tail = eclose.iloc[-self.vol_window - 1:]
            logr = np.log(tail.values[1:] / tail.values[:-1])
            rv = float(np.std(logr)) * np.sqrt(252)
            if rv < 1e-6:
                return []
            raw_exp = self.target_vol / rv
            exposure = min(self.max_exposure, max(self.min_exposure, raw_exp))
            eq_p = live_dict.get(_EQUITY, 0.0)
            if eq_p > 0:
                target[_EQUITY] = int(equity * exposure / eq_p)

        for sym, tgt_shares in list(target.items()):
            cur = int(ctx.position(sym).size)
            price = live_dict.get(sym, 0.0)
            if price > 0 and equity > 0:
                tw = tgt_shares * price / equity
                cw = cur * price / equity
                if abs(tw - cw) < self.rebalance_threshold:
                    target[sym] = cur

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size > 0:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        for sym, tgt in target.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > 0:
                orders.append(Order(side=OrderSide.BUY, size=float(delta), symbol=sym))
            elif delta < 0:
                orders.append(Order(side=OrderSide.SELL, size=float(-delta), symbol=sym))
        return orders


NAME = "opus1_qqq_asym_voltarget"
HYPOTHESIS = (
    "Mutate spy_vol_target_trend to RSP (equal-weight S&P) with 50d/150d MA "
    "cross gate + 60d-vol 11%-target; defensive bucket AGG; min 20% / max "
    "1.5x exposure. Equal-weight + cross-MA gate + AGG = different daily path."
)

STRATEGY = SpyCrossMAVolTarget()
