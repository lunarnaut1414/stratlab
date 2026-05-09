from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stratlab.engine.broker import Order
    from stratlab.engine.context import BarContext


class Strategy(ABC):
    """Base class for all strategies.

    Subclass and implement :meth:`on_bar` to return a list of orders (or empty
    list to do nothing). Use ``self.params`` to store tunable parameters.

    Strategies receive a :class:`BarContext` rather than raw frames. The
    context's ``history()`` accessor returns bars sliced to the current index
    inclusive, so look-ahead bias is structurally prevented for any strategy
    that uses it. For cross-sectional logic, use ``ctx.closes()`` /
    ``ctx.closes_window()`` to see all symbols at once.
    """

    def __init__(self, **params: float) -> None:
        self.params = params

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Called on each bar. Return orders to execute."""
        ...

    def on_start(self) -> None:
        """Called before the backtest begins. Override for initialization."""

    def on_end(self) -> None:
        """Called after the backtest finishes. Override for cleanup."""
