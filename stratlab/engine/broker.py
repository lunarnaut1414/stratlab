from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A trading order.

    ``limit_price`` controls execution semantics:

    - ``None`` (default): market order. Fills unconditionally at the
      current bar's open, with slippage applied (buys pay slightly more,
      sells receive slightly less).
    - Set to a float: limit order. Fills only if the current bar's
      OHLC range crosses the limit:
        - BUY:  fills when ``low <= limit_price``; fill price is
          ``min(limit_price, open)`` so a gap-down gives the better
          gap price (price improvement).
        - SELL: fills when ``high >= limit_price``; fill price is
          ``max(limit_price, open)`` so a gap-up gives the better
          gap price.
      Slippage is **not** applied to limit fills — the price you
      asked for is what you got.

    If a limit order's range condition isn't met, the order is dropped
    (counted in ``metrics["dropped_orders"]``).
    """
    side: OrderSide
    size: float
    symbol: str = ""
    limit_price: float | None = None


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


_NON_TRADEABLE_PREFIXES: tuple[str, ...] = ("^",)
_NON_TRADEABLE_SUFFIXES: tuple[str, ...] = ("=F", "=X")


def is_tradeable_symbol(symbol: str) -> bool:
    """Whether ``symbol`` names a real, directly-tradeable instrument.

    Cash equities, ETFs, ETNs, and ADRs all pass. The following Yahoo
    conventions are flagged as **signal-only** and rejected by the broker:

    - **``^...``** — index *levels* (``^VIX``, ``^TNX``, ``^GSPC``). You can't
      buy an index; route exposure through a futures contract or an
      index-tracking ETF (SPY, QQQ, etc.).
    - **``...=F``** — Yahoo's continuous, back-adjusted futures series
      (``ES=F``, ``GC=F``, ``CL=F``). The series exists for charting but is
      synthetic — there's no real contract that trades that price stream.
    - **``...=X``** — spot FX pairs (``EURUSD=X``). Use a currency ETF
      (``FXE``, ``UUP``) instead.

    Strategies can still *read* these symbols via ``ctx.history(sym)`` —
    they're useful as macro/regime signals — they just can't be ordered.
    """
    if not symbol:
        return False
    if symbol.startswith(_NON_TRADEABLE_PREFIXES):
        return False
    if symbol.endswith(_NON_TRADEABLE_SUFFIXES):
        return False
    return True


@dataclass
class Broker:
    """Simulated broker with long & short support.

    Orders flow through a single signed-size model: BUY adds to ``pos.size``,
    SELL subtracts. Crossing zero is allowed in a single fill (e.g. SELL 150 on
    a long-100 position flips to short-50). ``allow_short=False`` rejects orders
    that would push a position negative.

    **Cash gate**: when ``enforce_cash=True`` (default), BUY orders that would
    push cash below 0 are rejected (``fill_order`` returns ``None``). Set
    ``enforce_cash=False`` for legacy backtests that assume implicit margin.
    Note that going short still credits cash in this simplified model — to
    truly disable margin-via-shorts, pair ``enforce_cash=True`` with
    ``allow_short=False``.

    Borrow cost is modeled at the engine level via ``borrow_rate_annual`` and
    accrued daily on the absolute short notional.
    """

    initial_cash: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    allow_short: bool = True
    enforce_cash: bool = True
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
    ) -> Fill | None:
        """Try to fill ``order`` against ``bar`` (the current bar's OHLC).

        Returns the resulting :class:`Fill` if the order executed, or
        ``None`` if it didn't (limit not crossed, would push past the
        ``allow_short`` rule, or any required price column is NaN).

        Market orders fill at ``bar.open`` with slippage. Limit orders
        fill only if ``bar``'s low/high range crosses the limit and use
        the limit price (with gap protection — see :class:`Order`).
        """
        if not is_tradeable_symbol(order.symbol):
            return None  # index level / continuous future / FX — signal only

        open_raw = bar.get("open")
        if pd.isna(open_raw):
            return None
        open_raw = float(open_raw)

        if order.limit_price is None:
            # Market order: fill at open with slippage.
            if order.side == OrderSide.BUY:
                price = open_raw * (1 + self.slippage_pct)
            else:
                price = open_raw * (1 - self.slippage_pct)
        else:
            # Limit order: needs the high/low range to cross the limit.
            low_raw = bar.get("low")
            high_raw = bar.get("high")
            if pd.isna(low_raw) or pd.isna(high_raw):
                return None
            low = float(low_raw)
            high = float(high_raw)
            limit = float(order.limit_price)
            if order.side == OrderSide.BUY:
                if low > limit:
                    return None  # market never traded down to the limit
                # Gap protection: if the bar opened below our buy limit,
                # we get the better open price, not the limit price.
                price = min(limit, open_raw)
            else:
                if high < limit:
                    return None  # market never traded up to the limit
                price = max(limit, open_raw)

        signed = order.size if order.side == OrderSide.BUY else -order.size
        notional = abs(signed) * price
        commission = notional * self.commission_pct

        pos = self.get_position(order.symbol)
        old_size = pos.size
        new_size = old_size + signed

        if not self.allow_short and new_size < 0:
            return None

        # Cash gate: BUYs that would push cash negative are rejected.
        # SELLs of existing longs free cash; SELLs going short credit cash in
        # this simplified (margin-less) model — pair with allow_short=False
        # to remove the short-as-leverage loophole.
        if self.enforce_cash and order.side == OrderSide.BUY:
            cost = notional + commission
            if cost > self.cash:
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
