"""Limit-order fill semantics under the same-bar execution model.

Limit orders fill iff today's high/low range crosses the limit. The
fill price is the limit (with gap protection: a gap below a buy limit
gives the better gap-open price; a gap above a sell limit gives the
better open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratlab.engine.backtest import Backtest
from stratlab.engine.broker import Order, OrderSide
from stratlab.strategies.base import Strategy


def _bar_frame(rows: list[dict]) -> pd.DataFrame:
    """Build OHLCV from explicit per-bar dicts; volume defaulted."""
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    df = pd.DataFrame(rows, index=idx)
    if "volume" not in df.columns:
        df["volume"] = 1e6
    return df


class _LimitOnce(Strategy):
    """Submit a single limit order on bar 0."""
    def __init__(self, side: OrderSide, size: float, limit_price: float):
        super().__init__(side=side.value, size=size, limit_price=limit_price)
        self.side = side
        self.size = size
        self.limit_price = limit_price
        self.fired = False

    def on_bar(self, ctx):
        if self.fired:
            return []
        self.fired = True
        return [Order(side=self.side, size=self.size, limit_price=self.limit_price)]


def test_buy_limit_fills_when_low_touches_limit():
    """Bar low is below the buy limit → fill at the limit price."""
    df = _bar_frame([
        {"open": 105.0, "high": 110.0, "low": 99.0, "close": 108.0},  # low=99 ≤ limit=100
    ] + [{"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0}] * 5)
    result = Backtest(
        data={"A": df},
        strategy=_LimitOnce(OrderSide.BUY, size=1.0, limit_price=100.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    assert len(result.fills) == 1
    assert result.fills[0].price == pytest.approx(100.0)
    # Limit fills don't get slippage applied even if slippage_pct were nonzero.


def test_buy_limit_does_not_fill_when_low_above_limit():
    """Bar low never reaches the limit → order is dropped, no fill."""
    df = _bar_frame([
        {"open": 105.0, "high": 110.0, "low": 102.0, "close": 108.0},  # low=102 > limit=100
    ] + [{"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0}] * 5)
    result = Backtest(
        data={"A": df},
        strategy=_LimitOnce(OrderSide.BUY, size=1.0, limit_price=100.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    assert len(result.fills) == 0
    assert result.metrics["dropped_orders"] >= 1


def test_buy_limit_gap_down_gives_better_price():
    """Bar opens at $95 (below the $100 buy limit). We get filled at the open
    price, not the limit — gap protection."""
    df = _bar_frame([
        {"open": 95.0, "high": 105.0, "low": 94.0, "close": 100.0},
    ] + [{"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}] * 5)
    result = Backtest(
        data={"A": df},
        strategy=_LimitOnce(OrderSide.BUY, size=1.0, limit_price=100.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    assert len(result.fills) == 1
    assert result.fills[0].price == pytest.approx(95.0), \
        "gap-down should give buy a better fill than the limit"


def test_sell_limit_fills_when_high_touches_limit():
    """Bar high reaches above the sell limit → fill at the limit."""
    # Establish a long position first, then sell-limit it.
    class _BuyThenSellLimit(Strategy):
        def __init__(self):
            super().__init__()
            self.step = 0
        def on_bar(self, ctx):
            if self.step == 0:
                self.step = 1
                return [Order(side=OrderSide.BUY, size=1.0)]  # market
            if self.step == 1:
                self.step = 2
                return [Order(side=OrderSide.SELL, size=1.0, limit_price=110.0)]
            return []

    df = _bar_frame([
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"open": 105.0, "high": 112.0, "low": 104.0, "close": 108.0},  # high=112 ≥ 110
    ] + [{"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0}] * 3)
    result = Backtest(
        data={"A": df}, strategy=_BuyThenSellLimit(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    sell = [f for f in result.fills if f.side == OrderSide.SELL][0]
    assert sell.price == pytest.approx(110.0)


def test_sell_limit_does_not_fill_when_high_below_limit():
    """Bar high never reaches the limit → SELL is dropped, position unchanged."""
    class _BuyThenSellLimit(Strategy):
        def __init__(self):
            super().__init__()
            self.step = 0
        def on_bar(self, ctx):
            if self.step == 0:
                self.step = 1
                return [Order(side=OrderSide.BUY, size=1.0)]
            if self.step == 1:
                self.step = 2
                return [Order(side=OrderSide.SELL, size=1.0, limit_price=200.0)]
            return []

    df = _bar_frame([
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"open": 105.0, "high": 108.0, "low": 104.0, "close": 107.0},
    ] + [{"open": 107.0, "high": 110.0, "low": 106.0, "close": 109.0}] * 3)
    result = Backtest(
        data={"A": df}, strategy=_BuyThenSellLimit(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    sells = [f for f in result.fills if f.side == OrderSide.SELL]
    assert sells == []
    # Position should still be long 1 share at end.
    assert len(result.fills) == 1  # only the BUY


def test_sell_limit_gap_up_gives_better_price():
    """Bar opens at $115 (above the $110 sell limit). We get filled at the
    open price (115), not the limit — gap protection."""
    class _BuyThenSellLimit(Strategy):
        def __init__(self):
            super().__init__()
            self.step = 0
        def on_bar(self, ctx):
            if self.step == 0:
                self.step = 1
                return [Order(side=OrderSide.BUY, size=1.0)]
            if self.step == 1:
                self.step = 2
                return [Order(side=OrderSide.SELL, size=1.0, limit_price=110.0)]
            return []

    df = _bar_frame([
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        {"open": 115.0, "high": 120.0, "low": 114.0, "close": 118.0},
    ] + [{"open": 118.0, "high": 120.0, "low": 117.0, "close": 119.0}] * 3)
    result = Backtest(
        data={"A": df}, strategy=_BuyThenSellLimit(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    sell = [f for f in result.fills if f.side == OrderSide.SELL][0]
    assert sell.price == pytest.approx(115.0), \
        "gap-up should give sell a better fill than the limit"


def test_limit_orders_do_not_apply_slippage():
    """Limit fills use the asked price (or gap), not slippage-adjusted price."""
    df = _bar_frame([
        {"open": 105.0, "high": 110.0, "low": 99.0, "close": 108.0},
    ] + [{"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0}] * 5)
    result = Backtest(
        data={"A": df},
        strategy=_LimitOnce(OrderSide.BUY, size=1.0, limit_price=100.0),
        initial_cash=10_000.0, commission_pct=0.0,
        slippage_pct=0.01,  # would be 1% — but should NOT apply to limit
    ).run()
    assert len(result.fills) == 1
    assert result.fills[0].price == pytest.approx(100.0)


def test_paired_limits_can_round_trip_in_one_bar():
    """If a strategy submits a buy limit AND a sell limit on the same bar,
    and today's range crosses both, both fill — a same-day round-trip."""
    class _PairedOnce(Strategy):
        def __init__(self):
            super().__init__()
            self.fired = False
        def on_bar(self, ctx):
            if self.fired:
                return []
            self.fired = True
            return [
                Order(side=OrderSide.BUY, size=1.0, limit_price=100.0),
                Order(side=OrderSide.SELL, size=1.0, limit_price=110.0),
            ]

    # Bar low=98 (≤ buy limit 100), high=112 (≥ sell limit 110).
    df = _bar_frame([
        {"open": 105.0, "high": 112.0, "low": 98.0, "close": 108.0},
    ] + [{"open": 108.0, "high": 110.0, "low": 107.0, "close": 109.0}] * 3)
    result = Backtest(
        data={"A": df}, strategy=_PairedOnce(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    ).run()
    assert len(result.fills) == 2
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(110.0)
    assert trade.gross_pnl == pytest.approx(10.0)
    # Both fills happened on the same bar (same calendar day).
    assert trade.entry_time == trade.exit_time == df.index[0]
