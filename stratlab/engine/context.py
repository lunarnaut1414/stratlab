from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stratlab.engine.broker import Broker, Position


@dataclass
class BarContext:
    """State visible to a Strategy on a single bar.

    All history accessors return frames sliced to ``[0, idx]`` inclusive — never
    future bars — so strategies can't accidentally peek. Cross-sectional helpers
    like :meth:`closes` give the current bar across the whole tradeable universe
    in one call.
    """

    idx: int
    timestamp: pd.Timestamp
    symbols: list[str]
    _aligned: dict[str, pd.DataFrame] = field(repr=False)
    _closes_df: pd.DataFrame = field(repr=False)
    _broker: Broker = field(repr=False)

    def history(self, symbol: str | None = None) -> pd.DataFrame:
        """Bars [0, idx] inclusive for ``symbol``.

        If ``symbol`` is omitted, defaults to the first symbol — convenient for
        single-asset strategies. The returned frame is already truncated to
        prevent look-ahead bias.
        """
        sym = symbol if symbol is not None else next(iter(self._aligned))
        return self._aligned[sym].iloc[: self.idx + 1]

    def bar(self, symbol: str | None = None) -> pd.Series:
        """The OHLCV row at the current bar for ``symbol`` (default: first symbol)."""
        sym = symbol if symbol is not None else next(iter(self._aligned))
        return self._aligned[sym].iloc[self.idx]

    def closes(self) -> pd.Series:
        """Current close for every tradeable symbol at this bar (NaNs dropped).

        Use this for cross-sectional ranking, e.g.::

            ranked = ctx.closes().pct_change(periods=252).sort_values()
        """
        return self._closes_df.iloc[self.idx].dropna()

    def closes_window(self, lookback: int) -> pd.DataFrame:
        """Wide ``(lookback, n_symbols)`` close-price frame ending at the current bar.

        Useful for cross-sectional momentum / mean-reversion signals — compute
        returns or rolling stats column-wise, then rank.
        """
        start = max(0, self.idx - lookback + 1)
        return self._closes_df.iloc[start : self.idx + 1]

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
