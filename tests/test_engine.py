"""Backtest engine invariants.

These exercise the core guarantees the engine is supposed to provide:
- no look-ahead (orders fill at the *next* bar's open)
- cash conservation (no trade ⇒ cash unchanged)
- correct round-trip PnL accounting
- short-selling sign conventions
- commission and slippage applied as documented
- cross-sectional alignment with NaN-tolerance for not-yet-listed names

If any of these fail, every backtest result downstream is suspect.
"""
from __future__ import annotations

import math

import pytest

from stratlab.engine.backtest import Backtest
from stratlab.engine.broker import Order, OrderSide
from stratlab.strategies.base import Strategy


class _NoOp(Strategy):
    """Never trades."""
    def on_bar(self, ctx):
        return []


class _BuyOnce(Strategy):
    """Buy `size` shares of the (only) symbol on bar 0; hold forever."""
    def __init__(self, size: float = 10.0):
        super().__init__(size=size)
        self.size = size
        self.fired = False

    def on_bar(self, ctx):
        if self.fired:
            return []
        self.fired = True
        return [Order(side=OrderSide.BUY, size=self.size)]


class _RoundTrip(Strategy):
    """Buy on bar `entry_idx`, sell on `exit_idx`."""
    def __init__(self, entry_idx: int, exit_idx: int, size: float = 10.0):
        super().__init__(entry_idx=entry_idx, exit_idx=exit_idx, size=size)
        self.entry_idx = entry_idx
        self.exit_idx = exit_idx
        self.size = size

    def on_bar(self, ctx):
        if ctx.idx == self.entry_idx:
            return [Order(side=OrderSide.BUY, size=self.size)]
        if ctx.idx == self.exit_idx:
            return [Order(side=OrderSide.SELL, size=self.size)]
        return []


class _ShortOnce(Strategy):
    """Open a short on bar 0."""
    def __init__(self, size: float = 10.0):
        super().__init__(size=size)
        self.size = size
        self.fired = False

    def on_bar(self, ctx):
        if self.fired:
            return []
        self.fired = True
        return [Order(side=OrderSide.SELL, size=self.size)]


# -----------------------------------------------------------------------------


def test_no_trade_preserves_cash(flat_price):
    """If the strategy never trades, equity must equal initial_cash on every bar."""
    bt = Backtest(
        data={"A": flat_price}, strategy=_NoOp(),
        initial_cash=100_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert (result.equity_curve == 100_000.0).all(), \
        "no-op strategy somehow moved cash"
    assert len(result.fills) == 0
    assert len(result.trades) == 0


def test_orders_fill_at_next_bar_open(linear_ramp):
    """Order submitted on bar i must fill at bar (i+1)'s open price.

    With ramp prices 100, 101, 102, ..., a buy order placed by on_bar at idx=0
    must fill at price 101 (open of bar 1), not 100 (close of bar 0). Slippage
    off so the fill price equals the open exactly.
    """
    bt = Backtest(
        data={"A": linear_ramp}, strategy=_BuyOnce(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.price == pytest.approx(101.0), \
        f"expected fill at next-bar open (101.0), got {fill.price}"
    assert fill.timestamp == linear_ramp.index[1]


def test_final_bar_orders_dropped(linear_ramp):
    """An order placed on the last bar has no next bar to fill on — must be dropped."""
    last_idx = len(linear_ramp) - 1
    bt = Backtest(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=last_idx, exit_idx=last_idx),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert result.metrics.get("dropped_orders", 0) >= 1, \
        "order on final bar was filled — that's look-ahead/teleport-fill"


def test_round_trip_pnl_matches_price_diff(linear_ramp):
    """Buy at idx 0, sell at idx 50 → fills at next-bar opens (101 and 151).

    Gross PnL must equal (151 - 101) * 1 = 50, exactly.
    """
    bt = Backtest(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=0, exit_idx=50, size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(101.0)
    assert trade.exit_price == pytest.approx(151.0)
    assert trade.gross_pnl == pytest.approx(50.0)
    assert trade.side == "long"


def test_short_position_sign_and_cash(linear_ramp):
    """Sell-first opens a short: trade.side == 'short' and equity reflects price drop."""
    # Short on bar 0 (fills at bar 1, price 101), close on bar 50 (fills at bar 51, price 151).
    class _ShortRoundTrip(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                return [Order(side=OrderSide.SELL, size=2.0)]
            if ctx.idx == 50:
                return [Order(side=OrderSide.BUY, size=2.0)]  # cover
            return []

    bt = Backtest(
        data={"A": linear_ramp}, strategy=_ShortRoundTrip(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
        allow_short=True, borrow_rate_annual=0.0,
    )
    result = bt.run()
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short", f"expected short trade, got {trade.side}"
    # Shorting at 101 and covering at 151 on a rising market → loss of (151-101)*2 = 100.
    assert trade.gross_pnl == pytest.approx(-100.0), \
        f"short on rising market should lose: got pnl={trade.gross_pnl}"


def test_commission_deducted_per_fill(linear_ramp):
    """Same trade, two engines: with commission vs without. The diff is the commission."""
    common = dict(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=0, exit_idx=50, size=1.0),
        initial_cash=10_000.0, slippage_pct=0.0,
    )
    free = Backtest(commission_pct=0.0, **common).run()
    paid = Backtest(commission_pct=0.001, **common).run()  # 10 bps
    diff = free.equity_curve.iloc[-1] - paid.equity_curve.iloc[-1]
    # Two fills: buy at 101, sell at 151. Commission is 10bps of notional per fill.
    expected = 0.001 * (101.0 + 151.0)
    assert diff == pytest.approx(expected, rel=1e-6), \
        f"commission delta wrong: got {diff}, expected {expected}"


def test_slippage_buy_pays_more_sell_gets_less(linear_ramp):
    """With positive slippage_pct, buys fill above the open, sells fill below it."""
    bt = Backtest(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=0, exit_idx=50, size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.005,  # 50 bps
    )
    result = bt.run()
    buy_fill, sell_fill = result.fills
    # Buy fill: open 101, +50bps → 101 * 1.005 = 101.505
    assert buy_fill.price == pytest.approx(101.0 * 1.005)
    # Sell fill: open 151, -50bps → 151 * 0.995 = 150.245
    assert sell_fill.price == pytest.approx(151.0 * 0.995)
    # Net round-trip is worse than the no-slippage 50.0
    assert result.trades[0].gross_pnl < 50.0


def test_cross_sectional_alignment_handles_late_listing(two_assets):
    """B starts later than A. Engine should align on the union of indices and
    treat B as 'untradeable' before its listing date — no spurious orders or NaN-driven
    crashes."""
    bt = Backtest(
        data=two_assets, strategy=_NoOp(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    # Equity curve should span the full union, not just the overlap.
    expected_len = len(two_assets["A"].index.union(two_assets["B"].index))
    assert len(result.equity_curve) == expected_len
    assert (result.equity_curve == 10_000.0).all()


def test_buy_and_hold_tracks_price_change(linear_ramp):
    """Buy 1 share, hold to end. Final equity ≈ initial_cash + (last_price - entry_price)."""
    bt = Backtest(
        data={"A": linear_ramp}, strategy=_BuyOnce(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    last_price = float(linear_ramp["close"].iloc[-1])
    expected = 10_000.0 + (last_price - 101.0)  # bought at bar-1 open (101)
    assert result.equity_curve.iloc[-1] == pytest.approx(expected, rel=1e-9), \
        f"buy-and-hold PnL wrong: got {result.equity_curve.iloc[-1]}, expected {expected}"


def test_returns_consistent_with_equity_curve(linear_ramp):
    """returns.iloc[i] must equal pct change of equity from i-1 to i."""
    bt = Backtest(
        data={"A": linear_ramp}, strategy=_BuyOnce(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    eq = result.equity_curve
    expected = eq.pct_change().fillna(0.0)
    # Some engines may store returns slightly differently — allow tiny tolerance.
    aligned = result.returns.reindex(expected.index)
    assert ((aligned - expected).abs() < 1e-9).all(), \
        "returns series doesn't match equity_curve.pct_change()"
