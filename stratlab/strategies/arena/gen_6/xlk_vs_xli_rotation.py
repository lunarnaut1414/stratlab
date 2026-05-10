"""XLK vs XLI binary momentum rotation.

Hypothesis: Hold XLK (technology) or XLI (industrials), whichever has the
higher 60-day total return. 5-bar minimum hold. 97% concentration in winner.

Rationale:
- XLK leads in tech-driven, low-inflation environments
- XLI leads in manufacturing/infrastructure cycles, early recovery periods
- Both are all-equity positions -- no bond/cash drag
- 2010-2018: XLK generally wins, but XLI had 2010-2011, 2013-2014 leadership

Structural distinctions:
- All-equity, no bond/defensive rotation
- Different pair than XLK/XLU (no rate-sensitive utility) or QQQ/XLV
- Tech vs industrial cycle captures different macro signal
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

_ETF_A = "XLK"
_ETF_B = "XLI"
MOMENTUM_WINDOW = 60
MIN_HOLD_BARS = 5
EXPOSURE = 0.97


class XLKvsXLIRotation(Strategy):
    """Rank XLK vs XLI by 60d momentum; hold the stronger ETF fully."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        min_hold_bars: int = MIN_HOLD_BARS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            min_hold_bars=min_hold_bars,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.min_hold_bars = int(min_hold_bars)
        self.exposure = float(exposure)
        self._current_holding: str | None = None
        self._bars_since_switch: int = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        try:
            hist_a = ctx.history(_ETF_A)
            hist_b = ctx.history(_ETF_B)
        except Exception:
            return []

        if len(hist_a) < self.momentum_window + 1 or len(hist_b) < self.momentum_window + 1:
            return []

        closes = ctx.closes()
        if _ETF_A not in closes or _ETF_B not in closes:
            return []

        price_a = float(closes[_ETF_A])
        price_b = float(closes[_ETF_B])
        if price_a <= 0 or price_b <= 0:
            return []

        close_a = hist_a["close"].dropna()
        close_b = hist_b["close"].dropna()

        if len(close_a) < self.momentum_window or len(close_b) < self.momentum_window:
            return []

        start_a = float(close_a.iloc[-self.momentum_window])
        start_b = float(close_b.iloc[-self.momentum_window])

        if start_a <= 0 or start_b <= 0:
            return []

        mom_a = float(close_a.iloc[-1]) / start_a - 1.0
        mom_b = float(close_b.iloc[-1]) / start_b - 1.0

        target_sym = _ETF_A if mom_a >= mom_b else _ETF_B
        target_price = price_a if target_sym == _ETF_A else price_b

        self._bars_since_switch += 1

        if (self._current_holding is not None
                and self._current_holding != target_sym
                and self._bars_since_switch < self.min_hold_bars):
            return []

        if self._current_holding == target_sym:
            current_pos = ctx.position(target_sym)
            if current_pos.size > 0:
                return []

        orders: list[Order] = []

        for sym in [_ETF_A, _ETF_B]:
            pos = ctx.position(sym)
            if pos.size > 0 and sym != target_sym:
                orders.append(Order(side=OrderSide.SELL, size=pos.size, symbol=sym))

        equity = ctx.cash
        for sym in [_ETF_A, _ETF_B]:
            p = ctx.position(sym)
            if p.size > 0:
                price = price_a if sym == _ETF_A else price_b
                equity += p.size * price

        current = int(ctx.position(target_sym).size)
        target_shares = int(equity * self.exposure / target_price)
        if target_shares > current:
            delta = target_shares - current
            orders.append(Order(side=OrderSide.BUY, size=delta, symbol=target_sym))

        if orders:
            self._current_holding = target_sym
            self._bars_since_switch = 0

        return orders


UNIVERSE = ["XLK", "XLI"]
NAME = "xlk_vs_xli_rotation"
HYPOTHESIS = (
    "XLK vs XLI binary momentum rotation: hold whichever of XLK (tech) or XLI "
    "(industrials) has higher 60d return; 5-bar minimum hold; full concentration; "
    "no bond defensive"
)
STRATEGY = XLKvsXLIRotation()
