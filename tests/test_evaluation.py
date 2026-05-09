"""Tests for walk_forward + compare_to_benchmark.

These produce the numbers an agent uses to decide if a strategy is real
or overfit. If they're wrong, agents are wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratlab.engine.backtest import Backtest
from stratlab.engine.broker import Order, OrderSide
from stratlab.evaluation import compare_to_benchmark, walk_forward
from stratlab.strategies.base import Strategy


class _NoOp(Strategy):
    def on_bar(self, ctx):
        return []


class _BuyAndHold(Strategy):
    def __init__(self, size: float = 1.0):
        super().__init__(size=size)
        self.size = size
        self.fired = False

    def on_bar(self, ctx):
        if self.fired:
            return []
        self.fired = True
        return [Order(side=OrderSide.BUY, size=self.size)]


def _ohlcv(prices: np.ndarray, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": np.full_like(prices, 1e6, dtype=float)},
        index=idx,
    )


def test_walk_forward_partitions_into_expected_windows():
    """3 years of data with window_years=1 must produce 3 rows."""
    # 252 * 3 = 756 business days ≈ 3 years
    prices = np.linspace(100, 130, 756)
    data = {"A": _ohlcv(prices)}
    wf = walk_forward(_NoOp(), data, window_years=1.0)
    assert len(wf) == 3
    assert {"start", "end", "cagr", "sharpe", "max_drawdown"} <= set(wf.columns)


def test_walk_forward_raises_when_window_too_large_for_data():
    """1 year of data, window_years=5 → no windows fit. Must error helpfully."""
    prices = np.linspace(100, 110, 252)
    data = {"A": _ohlcv(prices)}
    with pytest.raises(ValueError, match="shorter than one window"):
        walk_forward(_NoOp(), data, window_years=5.0)


def test_walk_forward_no_op_strategy_yields_zero_returns():
    """No trades ⇒ flat equity ⇒ zero CAGR/Sharpe/vol per window."""
    prices = np.linspace(100, 130, 504)  # 2 years
    data = {"A": _ohlcv(prices)}
    wf = walk_forward(_NoOp(), data, window_years=1.0)
    assert (wf["cagr"].abs() < 1e-9).all()
    assert (wf["annual_volatility"].abs() < 1e-9).all()


def test_compare_to_benchmark_alpha_zero_when_strategy_equals_benchmark():
    """Buy-and-hold a single asset, then benchmark *against itself*. Alpha should be ~0
    for max_drawdown — both follow the same price path."""
    prices = np.linspace(100, 200, 252)
    df = _ohlcv(prices, start="2020-01-01")
    bench = pd.Series(prices, index=df.index)

    result = Backtest(
        data={"A": df}, strategy=_BuyAndHold(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()

    cmp = compare_to_benchmark(result, benchmark=bench)
    assert abs(cmp.loc["max_drawdown", "alpha"]) < 0.05


def test_compare_to_benchmark_returns_required_metrics():
    """Output frame must have strategy/benchmark/alpha cols and the standard metrics."""
    prices = np.linspace(100, 110, 252)
    df = _ohlcv(prices, start="2020-01-01")
    bench = pd.Series(prices, index=df.index)

    result = Backtest(
        data={"A": df}, strategy=_NoOp(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    cmp = compare_to_benchmark(result, benchmark=bench)

    assert list(cmp.columns) == ["strategy", "benchmark", "alpha"]
    assert {"cagr", "sharpe", "max_drawdown"} <= set(cmp.index)
    # Alpha column must equal strategy − benchmark by construction.
    assert ((cmp["alpha"] - (cmp["strategy"] - cmp["benchmark"])).abs() < 1e-9).all()


def test_compare_to_benchmark_handles_missing_overlap():
    """Benchmark series with no date overlap should raise rather than silently
    return garbage."""
    prices = np.linspace(100, 110, 60)
    df = _ohlcv(prices, start="2024-01-01")
    result = Backtest(
        data={"A": df}, strategy=_NoOp(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()

    # Benchmark on completely different dates
    far_future = pd.Series(
        np.linspace(100, 110, 30),
        index=pd.bdate_range("2030-01-01", periods=30),
    )
    with pytest.raises(ValueError, match="no overlap"):
        compare_to_benchmark(result, benchmark=far_future)
