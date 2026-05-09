from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stratlab.engine.broker import Broker, Position


@dataclass
class BarContext:
    """State visible to a Strategy on a single bar.

    The strategy decides at the *start* of bar ``idx`` — before today's
    OHLC has been observed. ``idx`` names the bar where any returned
    orders will execute, but the data accessors (:meth:`history`,
    :meth:`closes`, :meth:`closes_window`) only see bars **strictly
    before** ``idx``. This eliminates same-bar look-ahead by
    construction: there is no way to read today's close, high, or low
    from inside ``on_bar``.

    Cross-sectional helpers like :meth:`closes` and :meth:`closes_window`
    return the most recent observable closes — i.e., yesterday's closes —
    which is the correct basis for sizing today's orders.
    """

    idx: int
    timestamp: pd.Timestamp
    symbols: list[str]
    _aligned: dict[str, pd.DataFrame] = field(repr=False)
    _closes_df: pd.DataFrame = field(repr=False)
    _broker: Broker = field(repr=False)

    def history(self, symbol: str | None = None) -> pd.DataFrame:
        """Bars before today (``[0, idx)``) for ``symbol``.

        If ``symbol`` is omitted, defaults to the first symbol — convenient for
        single-asset strategies. The returned frame excludes today's bar so
        no indicator computed off it can leak into today's decision.
        """
        sym = symbol if symbol is not None else next(iter(self._aligned))
        return self._aligned[sym].iloc[: self.idx]

    def closes(self) -> pd.Series:
        """Most recent observable close for every tradeable symbol — i.e.,
        yesterday's closes (NaNs dropped).

        Use this for cross-sectional ranking and order sizing. On the very
        first bar (``idx == 0``) the result is empty: there's no prior bar
        to look at.
        """
        if self.idx == 0:
            return pd.Series(dtype=float)
        return self._closes_df.iloc[self.idx - 1].dropna()

    def closes_window(self, lookback: int) -> pd.DataFrame:
        """Wide ``(lookback, n_symbols)`` close-price frame ending at the
        most recent observable bar (yesterday).

        Useful for cross-sectional momentum / mean-reversion signals — compute
        returns or rolling stats column-wise, then rank.
        """
        end = self.idx  # exclusive — i.e., through yesterday inclusive
        start = max(0, end - lookback)
        return self._closes_df.iloc[start:end]

    def position(self, symbol: str) -> Position:
        return self._broker.get_position(symbol)

    @property
    def cash(self) -> float:
        return self._broker.cash

    @property
    def positions(self) -> dict[str, Position]:
        return {s: p for s, p in self._broker.positions.items() if p.size != 0}

    def portfolio_value(self, last_close: dict[str, float]) -> float:
        return self._broker.portfolio_value(last_close)
