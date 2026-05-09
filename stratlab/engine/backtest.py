from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stratlab.engine.broker import Broker
from stratlab.engine.context import BarContext

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
    """Backtest engine that runs a strategy over multi-asset OHLCV data.

    Symbols are aligned onto a common date index (the union of each frame's
    index). At each bar, only symbols with a valid close are considered
    *tradeable* — orders for not-yet-listed or delisted names are silently
    skipped. Held positions are marked-to-market at the last known close, so
    delisting doesn't vanish a position from the equity curve.

    Execution timing: orders submitted by ``on_bar`` on bar ``i`` are filled at
    bar ``i+1``'s open (with slippage). This prevents the look-ahead bias of
    deciding on a close and filling at that same close. Orders submitted on the
    final bar are dropped — surfaced as ``dropped_orders`` in the metrics dict.
    """

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

        if not self.data:
            raise ValueError("Backtest.data is empty")

        self.broker.reset()

        common_index = reduce(lambda a, b: a.union(b), (df.index for df in self.data.values()))
        common_index = common_index.sort_values()
        aligned = {sym: df.reindex(common_index) for sym, df in self.data.items()}
        closes_df = pd.DataFrame({sym: df["close"] for sym, df in aligned.items()})

        n_bars = len(common_index)
        equity = np.zeros(n_bars)
        last_close: dict[str, float] = {sym: 0.0 for sym in aligned}
        pending: list = []
        dropped_orders = 0

        self.strategy.on_start()

        for i in range(n_bars):
            bar_closes = closes_df.iloc[i]
            tradeable_mask = bar_closes.notna()
            tradeable = bar_closes.index[tradeable_mask].tolist()

            # Fill orders submitted on the previous bar at THIS bar's open.
            for order in pending:
                if order.symbol not in aligned:
                    dropped_orders += 1
                    continue
                bar = aligned[order.symbol].iloc[i]
                if pd.isna(bar.get("open")):
                    dropped_orders += 1
                    continue
                self.broker.fill_order(order, bar, common_index[i], price_col="open")
            pending = []

            for sym in tradeable:
                last_close[sym] = float(bar_closes[sym])

            ctx = BarContext(
                idx=i,
                timestamp=common_index[i],
                symbols=tradeable,
                _aligned=aligned,
                _closes_df=closes_df,
                _broker=self.broker,
            )

            orders = self.strategy.on_bar(ctx)

            for order in orders:
                if not order.symbol:
                    if not tradeable:
                        dropped_orders += 1
                        continue
                    order.symbol = tradeable[0]
                if order.symbol not in aligned:
                    dropped_orders += 1
                    continue
                pending.append(order)

            equity[i] = self.broker.portfolio_value(last_close)

        # Orders queued on the final bar can never fill — count and drop.
        dropped_orders += len(pending)

        self.strategy.on_end()

        equity_series = pd.Series(equity, index=common_index, name="equity")
        returns = equity_series.pct_change().fillna(0.0)
        metrics = compute_metrics(equity_series, returns)
        metrics["n_trades"] = len(self.broker.fills)
        metrics["dropped_orders"] = dropped_orders

        return BacktestResult(
            equity_curve=equity_series,
            returns=returns,
            fills=list(self.broker.fills),
            metrics=metrics,
        )
