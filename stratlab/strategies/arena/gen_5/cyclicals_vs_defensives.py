"""Cyclicals vs Defensives Composite Sector Rotation — gen_5 sonnet-9

Hypothesis:
  Use the composite 20-day return spread of cyclical sectors (XLI, XLF, XLK)
  vs defensive sectors (VPU, VDC) as a risk-on/risk-off barometer.

  - Risk-on regime: average 20-day return of cyclicals > defensives
    -> hold QQQ (growth proxy for cyclical expansion phase)
  - Risk-off regime: defensives leading
    -> hold TLT + IAU equally (bonds + gold safe haven)

  Rebalance weekly. XLI/XLF/XLK cyclicals vs VPU/VDC defensives provides a
  clean economic cycle signal without the noise of energy (XLE) which is
  driven by commodity supply shocks rather than pure economic cycle.

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Cyclical sectors: industrials, financials, technology
CYCLICAL_ETFS = ["XLI", "XLF", "XLK"]
# Defensive sectors: utilities, consumer staples
DEFENSIVE_ETFS = ["VPU", "VDC"]
# Risk-on asset
RISK_ON_ETF = "QQQ"
# Safe haven assets
SAFE_HAVEN_ETFS = ["TLT", "IAU"]

UNIVERSE = CYCLICAL_ETFS + DEFENSIVE_ETFS + [RISK_ON_ETF] + SAFE_HAVEN_ETFS

MOMENTUM_WINDOW = 15   # 15-day return spread for responsiveness
REBALANCE_DAYS = 5     # weekly rebalance
MIN_HISTORY = MOMENTUM_WINDOW + 10
EXPOSURE = 0.97


class CyclicalsVsDefensives(Strategy):
    """Cyclicals vs Defensives composite rotation: QQQ in risk-on, TLT+IAU in risk-off."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        rebalance_days: int = REBALANCE_DAYS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            rebalance_days=rebalance_days,
            exposure=exposure,
        )
        self.momentum_window = momentum_window
        self.rebalance_days = rebalance_days
        self.exposure = exposure
        self._bar_count: int = 0

    def on_start(self) -> None:
        self._bar_count = 0

    def _get_avg_return(self, ctx: BarContext, symbols: list[str]) -> tuple[float, int]:
        """Compute average N-day return across a list of ETFs available in context."""
        returns = []
        tradeable = set(ctx.symbols)
        for sym in symbols:
            if sym not in tradeable:
                continue
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            if len(hist) < self.momentum_window + 1:
                continue
            close = hist["close"]
            ret = float(close.iloc[-1] / close.iloc[-self.momentum_window] - 1.0)
            if np.isfinite(ret):
                returns.append(ret)
        if not returns:
            return 0.0, 0
        return float(np.mean(returns)), len(returns)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < MIN_HISTORY:
            return []

        self._bar_count += 1
        if self._bar_count % self.rebalance_days != 0:
            return []

        # Compute composite returns for each group
        cyclical_ret, n_cyclical = self._get_avg_return(ctx, CYCLICAL_ETFS)
        defensive_ret, n_defensive = self._get_avg_return(ctx, DEFENSIVE_ETFS)

        # Need at least 1 symbol from each group to make a valid comparison
        if n_cyclical == 0 or n_defensive == 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_closes = {s: float(closes[s]) for s in closes.index if closes[s] > 0}
        equity = ctx.portfolio_value(live_closes)
        if equity <= 0:
            return []

        # Determine regime
        risk_on = cyclical_ret > defensive_ret

        available = set(ctx.symbols)

        # Build target allocation
        if risk_on:
            if RISK_ON_ETF in available and live_closes.get(RISK_ON_ETF, 0) > 0:
                tradeable = [RISK_ON_ETF]
            else:
                # Fallback to cyclical basket
                tradeable = [s for s in CYCLICAL_ETFS if s in available and live_closes.get(s, 0) > 0]
        else:
            # Safe haven: TLT + IAU equally
            tradeable = [s for s in SAFE_HAVEN_ETFS if s in available and live_closes.get(s, 0) > 0]

        if not tradeable:
            return []

        per_weight = self.exposure / len(tradeable)
        target: dict[str, int] = {}
        for sym in tradeable:
            price = live_closes[sym]
            shares = int(equity * per_weight / price)
            if shares > 0:
                target[sym] = shares

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, tgt_shares in target.items():
            current = int(ctx.position(sym).size)
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "cyclicals_vs_defensives"
HYPOTHESIS = (
    "Cyclicals vs Defensives composite sector rotation: use (XLI+XLF+XLK) vs (VPU+VDC) "
    "15-day momentum spread as risk-on barometer; hold QQQ in risk-on expansion; "
    "rotate to TLT+IAU equally in risk-off contraction; rebalance weekly."
)

STRATEGY = CyclicalsVsDefensives()
