from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stratlab.engine.broker import Fill, OrderSide


@dataclass
class Trade:
    """A realized round-trip — entry to exit on a single symbol.

    Open positions at the end of the backtest are *not* emitted as trades; the
    caller can inspect ``broker.positions`` for those.
    """

    symbol: str
    side: str  # "long" | "short"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    return_pct: float


def extract_trades(fills: list[Fill]) -> list[Trade]:
    """Pair fills into round-trip trades, per symbol.

    Walks each symbol's fills in time order, tracking a signed running position
    and size-weighted entry price. Whenever the position is reduced or flipped,
    a Trade is emitted for the closed quantity. Flipping (e.g. SELL 150 from a
    long-100) emits a single trade for the closed 100 and leaves the residual
    50 as a fresh short whose exit happens on a later fill.
    """
    by_symbol: dict[str, list[Fill]] = defaultdict(list)
    for f in fills:
        by_symbol[f.symbol].append(f)

    trades: list[Trade] = []

    for sym, sym_fills in by_symbol.items():
        sym_fills.sort(key=lambda f: f.timestamp)

        size = 0.0  # signed running position
        avg = 0.0
        opened_at: pd.Timestamp | None = None

        for f in sym_fills:
            signed = f.size if f.side == OrderSide.BUY else -f.size

            if size == 0:
                size = signed
                avg = f.price
                opened_at = f.timestamp
                continue

            same_direction = (size > 0) == (signed > 0)

            if same_direction:
                new_size = size + signed
                avg = (avg * abs(size) + f.price * abs(signed)) / abs(new_size)
                size = new_size
                continue

            # opposite direction — close some/all of the position, possibly flipping
            close_qty = min(abs(size), abs(signed))
            if size > 0:
                pnl = close_qty * (f.price - avg)
                side_str = "long"
            else:
                pnl = close_qty * (avg - f.price)
                side_str = "short"

            trades.append(
                Trade(
                    symbol=sym,
                    side=side_str,
                    entry_time=opened_at,
                    exit_time=f.timestamp,
                    entry_price=avg,
                    exit_price=f.price,
                    size=close_qty,
                    gross_pnl=pnl,
                    return_pct=pnl / (close_qty * avg) if avg > 0 else 0.0,
                )
            )

            remainder = abs(signed) - abs(size)
            if remainder > 0:
                size = (1.0 if signed > 0 else -1.0) * remainder
                avg = f.price
                opened_at = f.timestamp
            elif remainder < 0:
                size = size + signed  # avg unchanged on a partial close
            else:
                size = 0.0
                avg = 0.0
                opened_at = None

    trades.sort(key=lambda t: t.exit_time)
    return trades


def trade_stats(trades: list[Trade]) -> dict[str, float]:
    """Aggregate stats over a list of round-trip trades."""
    if not trades:
        return {
            "n_round_trips": 0,
            "trade_win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_winner_pnl": 0.0,
            "avg_loser_pnl": 0.0,
            "avg_holding_days": 0.0,
            "avg_trade_return": 0.0,
        }

    pnls = np.array([t.gross_pnl for t in trades])
    rets = np.array([t.return_pct for t in trades])
    holdings = np.array([(t.exit_time - t.entry_time).days for t in trades])

    wins = pnls > 0
    win_pnl = pnls[wins].sum() if wins.any() else 0.0
    loss_pnl = -pnls[~wins].sum() if (~wins).any() else 0.0

    return {
        "n_round_trips": int(len(trades)),
        "trade_win_rate": round(float(wins.mean()), 4),
        "profit_factor": round(float(win_pnl / loss_pnl), 4) if loss_pnl > 0 else float("inf"),
        "avg_winner_pnl": round(float(pnls[wins].mean()), 2) if wins.any() else 0.0,
        "avg_loser_pnl": round(float(pnls[~wins].mean()), 2) if (~wins).any() else 0.0,
        "avg_holding_days": round(float(holdings.mean()), 2),
        "avg_trade_return": round(float(rets.mean()), 4),
    }


def annualized_turnover(fills: list[Fill], equity_curve: pd.Series) -> float:
    """Annualized two-way turnover: total notional traded / average equity, per year.

    A turnover of 1.0 means the portfolio churned through one full equity worth
    of notional per year. Long/short strategies often run 10x+.
    """
    if not fills or len(equity_curve) < 2:
        return 0.0
    total_notional = sum(f.price * f.size for f in fills)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = days / 365.25 if days > 0 else 1.0
    avg_equity = float(equity_curve.mean())
    if avg_equity <= 0:
        return 0.0
    return round((total_notional / years) / avg_equity, 4)
