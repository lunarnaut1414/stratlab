"""Backtest engine invariants under the same-bar limit-intraday model.

These exercise the core guarantees the engine is supposed to provide:

- on_bar runs *before* today's bar is observed; orders fill on the
  same bar's range
- market orders fill at today's open with slippage
- limit orders fill at the limit price (with gap protection) when the
  bar's low/high range crosses the limit; otherwise they're dropped
- cash conservation when the strategy doesn't trade
- correct round-trip PnL accounting
- short-selling sign conventions
- commission and slippage applied as documented
- cross-sectional alignment with NaN-tolerance for not-yet-listed names

If any of these fail, every backtest result downstream is suspect.
"""
from __future__ import annotations

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


def test_market_order_fills_at_current_bar_open(linear_ramp):
    """Market order submitted on bar i fills at bar i's open (same bar).

    With ramp prices 100, 101, 102, ..., a buy order placed on bar 0 must
    fill at price 100 (open of bar 0). Slippage off so fill price equals
    the open exactly.
    """
    bt = Backtest(
        data={"A": linear_ramp}, strategy=_BuyOnce(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.price == pytest.approx(100.0), \
        f"expected fill at current-bar open (100.0), got {fill.price}"
    assert fill.timestamp == linear_ramp.index[0]


def test_round_trip_pnl_matches_price_diff(linear_ramp):
    """Buy at idx 0, sell at idx 50 → fills at same-bar opens (100 and 150).

    Gross PnL must equal (150 - 100) * 1 = 50, exactly.
    """
    bt = Backtest(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=0, exit_idx=50, size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_price == pytest.approx(150.0)
    assert trade.gross_pnl == pytest.approx(50.0)
    assert trade.side == "long"


def test_short_round_trip_pnl(linear_ramp):
    """Open short on bar 0 (fills at 100), cover on bar 50 (fills at 150).

    Short on rising market → loss of (150 - 100) * 2 = 100.
    """
    class _ShortRoundTrip(Strategy):
        def on_bar(self, ctx):
            if ctx.idx == 0:
                return [Order(side=OrderSide.SELL, size=2.0)]
            if ctx.idx == 50:
                return [Order(side=OrderSide.BUY, size=2.0)]
            return []

    bt = Backtest(
        data={"A": linear_ramp}, strategy=_ShortRoundTrip(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
        allow_short=True, borrow_rate_annual=0.0,
    )
    result = bt.run()
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.side == "short"
    assert trade.gross_pnl == pytest.approx(-100.0)


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
    # Two fills: buy at 100, sell at 150. Commission is 10bps of notional per fill.
    expected = 0.001 * (100.0 + 150.0)
    assert diff == pytest.approx(expected, rel=1e-6), \
        f"commission delta wrong: got {diff}, expected {expected}"


def test_slippage_buy_pays_more_sell_gets_less(linear_ramp):
    """With positive slippage_pct, market buys fill above open, sells below."""
    bt = Backtest(
        data={"A": linear_ramp},
        strategy=_RoundTrip(entry_idx=0, exit_idx=50, size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.005,  # 50 bps
    )
    result = bt.run()
    buy_fill, sell_fill = result.fills
    # Buy at bar 0 open=100, +50bps → 100.5
    assert buy_fill.price == pytest.approx(100.0 * 1.005)
    # Sell at bar 50 open=150, -50bps → 149.25
    assert sell_fill.price == pytest.approx(150.0 * 0.995)
    assert result.trades[0].gross_pnl < 50.0


def test_cross_sectional_alignment_handles_late_listing(two_assets):
    """B starts later than A. Engine should align on the union of indices and
    treat B as 'untradeable' before its listing date — no spurious orders or
    NaN-driven crashes."""
    bt = Backtest(
        data=two_assets, strategy=_NoOp(),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    expected_len = len(two_assets["A"].index.union(two_assets["B"].index))
    assert len(result.equity_curve) == expected_len
    assert (result.equity_curve == 10_000.0).all()


def test_buy_and_hold_tracks_price_change(linear_ramp):
    """Buy 1 share on bar 0, hold to end. Final equity = initial_cash +
    (last_close - entry_price)."""
    bt = Backtest(
        data={"A": linear_ramp}, strategy=_BuyOnce(size=1.0),
        initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0,
    )
    result = bt.run()
    last_price = float(linear_ramp["close"].iloc[-1])
    expected = 10_000.0 + (last_price - 100.0)  # bought at bar-0 open
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
    aligned = result.returns.reindex(expected.index)
    assert ((aligned - expected).abs() < 1e-9).all(), \
        "returns series doesn't match equity_curve.pct_change()"


def test_history_excludes_today():
    """BarContext.history() must not include the current bar — that's the
    whole point of the same-bar fill model."""
    seen_lengths: list[int] = []
    seen_last_bar: list = []

    class _Probe(Strategy):
        def on_bar(self, ctx):
            seen_lengths.append(len(ctx.history()))
            if not ctx.history().empty:
                seen_last_bar.append(ctx.history().index[-1])
            return []

    import numpy as np
    import pandas as pd
    prices = np.arange(100.0, 105.0)
    df = pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [1e6] * 5},
        index=pd.bdate_range("2024-01-01", periods=5),
    )
    Backtest(data={"A": df}, strategy=_Probe(),
             initial_cash=10_000.0, commission_pct=0.0, slippage_pct=0.0).run()

    # On bar i, history() should have exactly i bars (bars 0..i-1, today excluded).
    assert seen_lengths == [0, 1, 2, 3, 4]
    # And the last visible bar is always strictly before today's index.
    assert seen_last_bar == list(df.index[:-1])
