"""Round-trip extraction — every metric an agent will judge a strategy by
flows through ``extract_trades``, so its edge cases need direct coverage."""
from __future__ import annotations

import pandas as pd
import pytest

from stratlab.analytics.trades import (
    annualized_turnover, extract_trades, trade_stats,
)
from stratlab.engine.broker import Fill, OrderSide


def _fill(symbol: str, side: OrderSide, size: float, price: float, day: int) -> Fill:
    return Fill(
        symbol=symbol, side=side, size=size, price=price,
        timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
    )


def test_extract_trades_simple_long_round_trip():
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.SELL, 10, 110.0, 5),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 1
    t = trades[0]
    assert t.symbol == "A"
    assert t.side == "long"
    assert t.size == 10
    assert t.entry_price == 100.0
    assert t.exit_price == 110.0
    assert t.gross_pnl == pytest.approx(100.0)  # (110 - 100) * 10
    assert t.return_pct == pytest.approx(0.1)


def test_extract_trades_short_round_trip():
    fills = [
        _fill("A", OrderSide.SELL, 5, 50.0, 0),  # open short at 50
        _fill("A", OrderSide.BUY,  5, 40.0, 3),  # cover at 40
    ]
    trades = extract_trades(fills)
    assert len(trades) == 1
    assert trades[0].side == "short"
    assert trades[0].gross_pnl == pytest.approx(50.0)  # (50 - 40) * 5


def test_extract_trades_flip_long_to_short():
    """SELL more than the long size flips into a short — should emit one closing
    trade for the original long, leave the residual as an open short."""
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.SELL, 15, 120.0, 5),  # close 10 long, open 5 short
        _fill("A", OrderSide.BUY,   5,  90.0, 10), # cover the residual short
    ]
    trades = extract_trades(fills)
    assert len(trades) == 2
    long_trade, short_trade = trades
    assert long_trade.side == "long"
    assert long_trade.size == 10
    assert long_trade.gross_pnl == pytest.approx(200.0)  # 10 * (120 - 100)
    assert short_trade.side == "short"
    assert short_trade.size == 5
    assert short_trade.gross_pnl == pytest.approx(150.0)  # 5 * (120 - 90)


def test_extract_trades_partial_close_keeps_avg_entry():
    """Partially closing a long reduces size but doesn't change avg_entry."""
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.SELL,  4, 110.0, 2),  # partial close
        _fill("A", OrderSide.SELL,  6, 120.0, 5),  # final close
    ]
    trades = extract_trades(fills)
    assert len(trades) == 2
    # both trades are at avg_entry=100 (unchanged across partial)
    assert trades[0].entry_price == pytest.approx(100.0)
    assert trades[1].entry_price == pytest.approx(100.0)
    total_pnl = sum(t.gross_pnl for t in trades)
    assert total_pnl == pytest.approx(40 + 120)  # 4*(110-100) + 6*(120-100)


def test_extract_trades_pyramiding_recomputes_avg_entry():
    """Adding to an existing long should size-weight the entry price."""
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.BUY,  10, 120.0, 2),  # adds, avg becomes 110
        _fill("A", OrderSide.SELL, 20, 130.0, 5),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(110.0)
    assert trades[0].gross_pnl == pytest.approx(20 * (130.0 - 110.0))


def test_extract_trades_multi_symbol_independence():
    """Fills for different symbols must not cross-contaminate."""
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("B", OrderSide.BUY,   5, 200.0, 0),
        _fill("A", OrderSide.SELL, 10, 105.0, 3),
        _fill("B", OrderSide.SELL,  5, 195.0, 4),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 2
    by_symbol = {t.symbol: t for t in trades}
    assert by_symbol["A"].gross_pnl == pytest.approx(50.0)
    assert by_symbol["B"].gross_pnl == pytest.approx(-25.0)


def test_extract_trades_open_position_not_emitted():
    """A still-open position at end-of-fills must not produce a phantom Trade."""
    fills = [_fill("A", OrderSide.BUY, 10, 100.0, 0)]
    trades = extract_trades(fills)
    assert trades == []


def test_trade_stats_handles_empty_list():
    s = trade_stats([])
    assert s["n_round_trips"] == 0
    assert s["trade_win_rate"] == 0.0


def test_trade_stats_profit_factor_is_inf_when_no_losers():
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.SELL, 10, 110.0, 5),
    ]
    stats = trade_stats(extract_trades(fills))
    assert stats["n_round_trips"] == 1
    assert stats["trade_win_rate"] == 1.0
    assert stats["profit_factor"] == float("inf")


def test_annualized_turnover_basic():
    fills = [
        _fill("A", OrderSide.BUY,  10, 100.0, 0),
        _fill("A", OrderSide.SELL, 10, 100.0, 365),
    ]
    eq = pd.Series(
        [10_000.0, 10_000.0],
        index=pd.to_datetime(["2024-01-01", "2025-01-01"]),
    )
    # total notional = 10*100 + 10*100 = 2000; avg equity = 10000;
    # 1 year → turnover = 2000/10000 = 0.2
    t = annualized_turnover(fills, eq)
    assert t == pytest.approx(0.2, abs=0.01)
