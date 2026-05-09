from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    side: OrderSide
    size: float
    symbol: str = ""


@dataclass
class Position:
    symbol: str
    size: float = 0.0
    avg_entry: float = 0.0


@dataclass
class Fill:
    symbol: str
    side: OrderSide
    size: float
    price: float
    timestamp: pd.Timestamp


@dataclass
class Broker:
    """Simulated broker that tracks positions, cash, and fills orders with configurable slippage."""

    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def reset(self) -> None:
        self.cash = self.initial_cash
        self.positions.clear()
        self.fills.clear()

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def fill_order(self, order: Order, bar: pd.Series, timestamp: pd.Timestamp) -> Fill | None:
        price = float(bar["close"])

        if order.side == OrderSide.BUY:
            price *= 1 + self.slippage_pct
        else:
            price *= 1 - self.slippage_pct

        cost = price * order.size
        commission = cost * self.commission_pct

        if order.side == OrderSide.BUY:
            if self.cash < cost + commission:
                return None
            self.cash -= cost + commission
            pos = self.get_position(order.symbol)
            total_cost = pos.avg_entry * pos.size + price * order.size
            pos.size += order.size
            pos.avg_entry = total_cost / pos.size if pos.size > 0 else 0.0
        else:
            pos = self.get_position(order.symbol)
            sell_size = min(order.size, pos.size)
            if sell_size <= 0:
                return None
            self.cash += price * sell_size - commission
            pos.size -= sell_size
            if pos.size == 0:
                pos.avg_entry = 0.0

        f = Fill(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            price=price,
            timestamp=timestamp,
        )
        self.fills.append(f)
        return f

    def portfolio_value(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            pos.size * prices.get(pos.symbol, 0.0) for pos in self.positions.values()
        )
        return self.cash + positions_value
