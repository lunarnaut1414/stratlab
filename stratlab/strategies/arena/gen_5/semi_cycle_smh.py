"""Semiconductor cycle timing using SMH vs SPY relative return.

Hypothesis:
  Semiconductors are a leading indicator of the broader tech cycle. When
  SMH's 3-month return exceeds SPY's 3-month return by >5 percentage points
  AND both have positive momentum, the semi cycle is in expansion. Hold SMH
  for concentrated tech exposure in this regime.

  Signal logic:
  - If SMH_3mo_return > SPY_3mo_return + threshold AND SMH_3mo_return > 0:
      HOLD SMH (semiconductor expansion regime)
  - Else:
      HOLD SHY (short-term Treasuries — cash proxy)

  Rebalance: weekly (every 5 trading days).
  Use SOXX as secondary confirmation: only enter if SOXX also has positive
  3-month return (avoids single-ETF data anomalies).

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SMH", "SOXX", "SPY", "SHY"]

_LOOKBACK = 63       # ~3 months (trading days)
_THRESHOLD = 0.05    # SMH must beat SPY by 5pp to enter
_REBALANCE = 5       # weekly rebalance
_EXPOSURE = 0.97
_MIN_HISTORY = _LOOKBACK + 10


class SemiCycleSMH(Strategy):
    """Semiconductor cycle regime timer: SMH vs SPY 3-month return spread.

    Parameters
    ----------
    lookback : int
        Return lookback in trading days (default 63 = ~3 months).
    threshold : float
        Required outperformance of SMH over SPY to enter semiconductor regime
        (default 0.05 = 5 percentage points).
    rebalance : int
        Bars between rebalance checks (default 5 = weekly).
    exposure : float
        Fraction of portfolio deployed (default 0.97).
    """

    def __init__(
        self,
        lookback: int = _LOOKBACK,
        threshold: float = _THRESHOLD,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(
            lookback=lookback,
            threshold=threshold,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.lookback = lookback
        self.threshold = threshold
        self.rebalance = rebalance
        self.exposure = exposure

    def _get_3mo_return(self, ctx: BarContext, symbol: str) -> float | None:
        """Compute lookback-period return for symbol. Returns None on failure."""
        try:
            hist = ctx.history(symbol)
        except Exception:
            return None
        if hist is None or len(hist) < self.lookback + 2:
            return None
        closes = hist["close"].dropna()
        if len(closes) < self.lookback + 1:
            return None
        past_price = float(closes.iloc[-(self.lookback + 1)])
        curr_price = float(closes.iloc[-1])
        if past_price <= 0:
            return None
        return (curr_price - past_price) / past_price

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < _MIN_HISTORY:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Compute 3-month returns for key symbols
        smh_ret = self._get_3mo_return(ctx, "SMH")
        spy_ret = self._get_3mo_return(ctx, "SPY")
        soxx_ret = self._get_3mo_return(ctx, "SOXX")

        if smh_ret is None or spy_ret is None:
            return []

        # Signal: SMH outperforms SPY by threshold AND both SMH and SOXX are positive
        semi_regime = (
            smh_ret > (spy_ret + self.threshold)
            and smh_ret > 0.0
            and (soxx_ret is None or soxx_ret > 0.0)  # SOXX confirmation (skip if unavailable)
        )

        target_sym = "SMH" if semi_regime else "SHY"
        exit_sym = "SHY" if semi_regime else "SMH"

        closes = ctx.closes()
        if target_sym not in closes.index:
            return []

        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit the position we're not targeting
        exit_pos = ctx.position(exit_sym)
        if exit_pos.size > 0:
            orders.append(Order(side=OrderSide.SELL, size=exit_pos.size, symbol=exit_sym))

        # Also exit SOXX if held
        soxx_pos = ctx.position("SOXX")
        if soxx_pos.size > 0:
            orders.append(Order(side=OrderSide.SELL, size=soxx_pos.size, symbol="SOXX"))
        spy_pos = ctx.position("SPY")
        if spy_pos.size > 0:
            orders.append(Order(side=OrderSide.SELL, size=spy_pos.size, symbol="SPY"))

        # Size target position
        target_price = live_closes_dict.get(target_sym)
        if not target_price or target_price <= 0:
            return orders

        target_shares = int(equity * self.exposure / target_price)
        current_shares = int(ctx.position(target_sym).size)
        delta = target_shares - current_shares

        if delta > 0:
            orders.append(Order(side=OrderSide.BUY, size=float(delta), symbol=target_sym))
        elif delta < -1:
            orders.append(Order(side=OrderSide.SELL, size=float(abs(delta)), symbol=target_sym))

        return orders


NAME = "semi_cycle_smh"
HYPOTHESIS = (
    "Semiconductor cycle timing: buy SMH when SMH 3-month return exceeds SPY "
    "3-month return by >5pp AND both are positive (SOXX secondary confirmation); "
    "rotate to SHY cash-proxy otherwise; weekly rebalance targets semi-cycle outperformance."
)

STRATEGY = SemiCycleSMH(
    lookback=63,
    threshold=0.05,
    rebalance=5,
    exposure=0.97,
)
