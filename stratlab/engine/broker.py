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
    """Holdings for one symbol.

    ``size`` is signed: positive = long, negative = short. ``avg_entry`` is the
    size-weighted average entry price for the *current* position (resets when
    the position crosses zero).
    """

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
    """Simulated broker with long & short support.

    Orders flow through a single signed-size model: BUY adds to ``pos.size``,
    SELL subtracts. Crossing zero is allowed in a single fill (e.g. SELL 150 on
    a long-100 position flips to short-50). ``allow_short=False`` rejects orders
    that would push a position negative.

    Margin is *not* enforced — strategies are responsible for sizing within
    cash + reasonable leverage. Borrow cost is modeled at the engine level via
    ``borrow_rate_annual`` and accrued daily on the absolute short notional.
    """

    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    allow_short: bool = True
    borrow_rate_annual: float = 0.0

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

    def fill_order(
        self,
        order: Order,
        bar: pd.Series,
        timestamp: pd.Timestamp,
        price_col: str = "close",
    ) -> Fill | None:
        raw = float(bar[price_col])
        price = raw * (1 + self.slippage_pct) if order.side == OrderSide.BUY else raw * (1 - self.slippage_pct)

        signed = order.size if order.side == OrderSide.BUY else -order.size
        notional = abs(signed) * price
        commission = notional * self.commission_pct

        pos = self.get_position(order.symbol)
        old_size = pos.size
        new_size = old_size + signed

        if not self.allow_short and new_size < 0:
            return None

        # Cash flow: BUY pays out, SELL receives proceeds. Commission always reduces cash.
        self.cash += -signed * price - commission

        # avg_entry update:
        # - extending in the same direction (or opening from flat): weighted avg
        # - reducing without crossing zero: avg_entry unchanged
        # - flat: reset to 0
        # - crossing zero: residual is a fresh position at the current price
        if old_size == 0:
            pos.avg_entry = price if new_size != 0 else 0.0
        elif _same_sign(old_size, new_size) and abs(new_size) > abs(old_size):
            pos.avg_entry = (
                pos.avg_entry * abs(old_size) + price * abs(signed)
            ) / abs(new_size)
        elif new_size == 0:
            pos.avg_entry = 0.0
        elif not _same_sign(old_size, new_size):
            pos.avg_entry = price
        # else: same sign, reducing — avg_entry unchanged

        pos.size = new_size

        f = Fill(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            price=price,
            timestamp=timestamp,
        )
        self.fills.append(f)
        return f

    def accrue_borrow(self, prices: dict[str, float], days: float = 1.0) -> float:
        """Charge borrow cost on short positions for ``days`` calendar days.

        Returns the total cost charged. No-op if ``borrow_rate_annual == 0``.
        """
        if self.borrow_rate_annual == 0.0:
            return 0.0
        rate_per_day = self.borrow_rate_annual / 365.0
        cost = 0.0
        for pos in self.positions.values():
            if pos.size < 0:
                price = prices.get(pos.symbol, 0.0)
                cost += abs(pos.size) * price * rate_per_day * days
        self.cash -= cost
        return cost

    def portfolio_value(self, prices: dict[str, float]) -> float:
        positions_value = sum(
            pos.size * prices.get(pos.symbol, 0.0) for pos in self.positions.values()
        )
        return self.cash + positions_value


def _same_sign(a: float, b: float) -> bool:
    return (a > 0 and b > 0) or (a < 0 and b < 0)
