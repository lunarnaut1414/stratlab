from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stratlab.engine.broker import Broker

if TYPE_CHECKING:
    from stratlab.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    fills: list
    metrics: dict[str, float]


@dataclass
class Backtest:
    """Vectorized backtest engine that runs a strategy over OHLCV data."""

    data: dict[str, pd.DataFrame]
    strategy: Strategy
    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

    broker: Broker = field(init=False)

    def __post_init__(self) -> None:
        self.broker = Broker(
            initial_cash=self.initial_cash,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )

    def run(self) -> BacktestResult:
        from stratlab.analytics.metrics import compute_metrics

        self.broker.reset()

        symbols = list(self.data.keys())
        primary = self.data[symbols[0]]
        n_bars = len(primary)

        equity = np.zeros(n_bars)
        dates = primary.index

        self.strategy.on_start()

        for i in range(n_bars):
            orders = self.strategy.on_bar(i, primary)

            for order in orders:
                if not order.symbol:
                    order.symbol = symbols[0]
                bar = self.data[order.symbol].iloc[i]
                self.broker.fill_order(order, bar, dates[i])

            prices = {}
            for sym in symbols:
                prices[sym] = float(self.data[sym].iloc[i]["close"])
            equity[i] = self.broker.portfolio_value(prices)

        self.strategy.on_end()

        equity_series = pd.Series(equity, index=dates, name="equity")
        returns = equity_series.pct_change().fillna(0.0)
        metrics = compute_metrics(equity_series, returns)
        metrics["n_trades"] = len(self.broker.fills)

        return BacktestResult(
            equity_curve=equity_series,
            returns=returns,
            fills=list(self.broker.fills),
            metrics=metrics,
        )
