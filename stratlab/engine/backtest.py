from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from stratlab.engine.broker import Broker
from stratlab.engine.context import BarContext

if TYPE_CHECKING:
    from stratlab.analytics.trades import Trade
    from stratlab.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    fills: list
    trades: list[Trade]
    metrics: dict[str, float]


@dataclass
class Backtest:
    """Backtest engine that runs a strategy over multi-asset OHLCV data.

    Symbols are aligned onto a common date index (the union of each frame's
    index). At each bar, only symbols with a valid close are considered
    *tradeable* — orders for not-yet-listed or delisted names are silently
    skipped. Held positions are marked-to-market at the last known close, so
    delisting doesn't vanish a position from the equity curve.

    Execution timing: ``on_bar`` runs *before* bar ``i`` is observed —
    ``ctx.history()`` returns bars [0, i) (yesterday and earlier). Orders
    returned by ``on_bar`` are checked against bar ``i``'s OHLC range:

    - Market orders (``limit_price=None``) fill at bar ``i``'s open with
      slippage applied (buys pay slightly more, sells receive slightly less).
    - Limit orders fill only if the bar's range crosses the limit:
      buys when ``low <= limit_price``, sells when ``high >= limit_price``.
      A gap below a buy limit (or above a sell limit) gives the better gap
      price. No slippage on limit fills.

    This eliminates look-ahead structurally — there's no way for the strategy
    to read today's close, high, or low when deciding. A paired BUY-limit
    and SELL-limit on the same bar can produce a same-bar round-trip when
    today's range crosses both.
    """

    data: dict[str, pd.DataFrame]
    strategy: Strategy
    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    allow_short: bool = True
    borrow_rate_annual: float = 0.0

    broker: Broker = field(init=False)

    def __post_init__(self) -> None:
        self.broker = Broker(
            initial_cash=self.initial_cash,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
            allow_short=self.allow_short,
            borrow_rate_annual=self.borrow_rate_annual,
        )

    def run(self) -> BacktestResult:
        from stratlab.analytics.metrics import compute_metrics
        from stratlab.analytics.trades import (
            annualized_turnover,
            extract_trades,
            trade_stats,
        )

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
        dropped_orders = 0
        total_borrow_cost = 0.0
        prev_ts: pd.Timestamp | None = None

        self.strategy.on_start()

        for i in range(n_bars):
            bar_closes = closes_df.iloc[i]
            tradeable = bar_closes.index[bar_closes.notna()].tolist()

            # Borrow accrual covers the calendar gap from the prior bar to this
            # bar's open, before today's fills happen.
            if prev_ts is not None and self.broker.borrow_rate_annual != 0.0:
                days = (common_index[i] - prev_ts).days
                total_borrow_cost += self.broker.accrue_borrow(last_close, days=days)
            prev_ts = common_index[i]

            # Strategy decides BEFORE today's bar is observed. ``history()``
            # returns bars [0, i) — yesterday and earlier. ``ctx.idx == i``
            # still names today (the bar where any orders will execute).
            ctx = BarContext(
                idx=i,
                timestamp=common_index[i],
                symbols=tradeable,
                _aligned=aligned,
                _closes_df=closes_df,
                _broker=self.broker,
            )
            orders = self.strategy.on_bar(ctx)

            # Try to fill each order against today's OHLC range.
            for order in orders:
                if not order.symbol:
                    if not tradeable:
                        dropped_orders += 1
                        continue
                    order.symbol = tradeable[0]
                if order.symbol not in aligned:
                    dropped_orders += 1
                    continue
                bar = aligned[order.symbol].iloc[i]
                fill = self.broker.fill_order(order, bar, common_index[i])
                if fill is None:
                    dropped_orders += 1

            # Now that fills (if any) have updated positions, mark to today's close.
            for sym in tradeable:
                last_close[sym] = float(bar_closes[sym])
            equity[i] = self.broker.portfolio_value(last_close)

        self.strategy.on_end()

        equity_series = pd.Series(equity, index=common_index, name="equity")
        returns = equity_series.pct_change().fillna(0.0)
        metrics = compute_metrics(equity_series, returns)

        fills = list(self.broker.fills)
        trades = extract_trades(fills)

        metrics["n_trades"] = len(fills)
        metrics["dropped_orders"] = dropped_orders
        metrics["borrow_cost"] = round(total_borrow_cost, 2)
        metrics["turnover_annualized"] = annualized_turnover(fills, equity_series)
        metrics.update(trade_stats(trades))

        return BacktestResult(
            equity_curve=equity_series,
            returns=returns,
            fills=fills,
            trades=trades,
            metrics=metrics,
        )
