from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from stratlab.engine.broker import Order


class Strategy(ABC):
    """Base class for all strategies.

    Subclass and implement `on_bar` to return a list of orders (or empty list to do nothing).
    Use `self.params` to store tunable parameters.
    """

    def __init__(self, **params: float) -> None:
        self.params = params

    @abstractmethod
    def on_bar(self, idx: int, history: pd.DataFrame) -> list[Order]:
        """Called on each bar. Return orders to execute.

        Args:
            idx: Current bar index (0-based). history[:idx+1] is visible data.
            history: Full OHLCV dataframe. Only use rows up to idx to avoid look-ahead bias.
        """
        ...

    def on_start(self) -> None:
        """Called before the backtest begins. Override for initialization."""

    def on_end(self) -> None:
        """Called after the backtest finishes. Override for cleanup."""
