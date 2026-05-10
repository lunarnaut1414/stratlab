"""opus-1 mutation of atr_momentum_etf (etf_rotation cluster, parent IS Calmar 0.74).

Structural mutations vs parent (RSP/SPY breadth -> ^MOVE bond-vol regime,
QQQ/SPY/TLT universe -> MTUM/QUAL/IEF/SPY universe):

  - Regime signal:   RSP/SPY ratio vs 40d MA -> ^MOVE (bond-vol index)
                                                 vs its own 60d MA. Bond
                                                 volatility is the driver of
                                                 cross-asset risk, uncorrelated
                                                 with VIX.
  - Universe:        QQQ / SPY / TLT  -> MTUM / QUAL / IEF / SPY (factor-tilt
                                          rotation instead of cap-weighted
                                          rotation).
  - Risk-on regime:  QQQ when breadth positive  -> MTUM when MOVE below MA
                                                    (calm bond market favors
                                                     momentum factor).
  - Risk-off:        SPY -> QUAL when MOVE above MA (high bond vol -> lean
                            into quality / low-leverage names).
  - Bear:            TLT when SPY<200d -> IEF when SPY<200d (mid-duration,
                                          less rates-sensitive).
  - Fallback before MTUM/QUAL exist (pre-2013-07): SPY when bull, IEF when
    bear. The harness aligns universes by overlap so the strategy still
    runs — fallback keeps it active during 2010-2013.

  - Rebalance:       10 bars -> 21 bars (monthly, less churn).

The MOVE/VIX/MTUM/QUAL combination is structurally novel vs the
RSP/SPY/QQQ/TLT path of the parent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "MTUM", "QUAL", "IEF", "^MOVE"]

MOVE_MA = 60
TREND_WINDOW = 200
REBALANCE = 21
EXPOSURE = 0.97


class MoveFactorRotation(Strategy):
    def __init__(
        self,
        move_ma: int = MOVE_MA,
        trend_window: int = TREND_WINDOW,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            move_ma=move_ma,
            trend_window=trend_window,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.move_ma = int(move_ma)
        self.trend_window = int(trend_window)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.move_ma, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        try:
            move_hist = ctx.history("^MOVE")
        except KeyError:
            move_hist = None

        if len(spy_hist) < self.trend_window + 5:
            return []

        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        # MOVE regime
        move_low = True  # default to "calm" if MOVE missing
        if move_hist is not None and len(move_hist) >= self.move_ma + 1:
            mc = move_hist["close"].dropna()
            move_now = float(mc.iloc[-1])
            move_ma = float(mc.iloc[-self.move_ma:].mean())
            if np.isfinite(move_now) and np.isfinite(move_ma):
                move_low = move_now < move_ma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target ticker.
        # Rules:
        #   bear (SPY<200d):       IEF
        #   bull & calm bonds:     MTUM (factor-momentum)  -- fall back to SPY
        #                            if MTUM not yet listed
        #   bull & vol bonds:      QUAL                    -- fall back to SPY
        if not bull:
            target_sym = "IEF"
        elif move_low:
            target_sym = "MTUM" if "MTUM" in closes_now.index else "SPY"
        else:
            target_sym = "QUAL" if "QUAL" in closes_now.index else "SPY"

        if target_sym not in closes_now.index:
            return []
        price = float(closes_now[target_sym])
        if price <= 0:
            return []

        target_shares = int(equity * self.exposure / price)
        orders: list[Order] = []

        # Exit other positions
        for sym, pos in list(ctx.positions.items()):
            if sym != target_sym and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        cur = int(ctx.position(target_sym).size)
        delta = target_shares - cur
        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=abs(delta), symbol=target_sym))
        elif delta < 0:
            orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=target_sym))

        return orders


NAME = "opus1_etf_move_factor_rotation"
HYPOTHESIS = (
    "Mutate atr_momentum_etf: swap regime to ^MOVE bond-vol — hold MTUM in "
    "low-MOVE, QUAL in high-MOVE, IEF when SPY<200d. Factor ETF rotation "
    "gated by bond vol. Pre-2013 fallback to SPY/IEF."
)

STRATEGY = MoveFactorRotation()
