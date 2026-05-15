"""SP500 breadth-timed QQQ/IEF allocation.

Hypothesis: SP500 internal breadth (fraction of stocks above 50d SMA) is a
leading indicator of market regime quality. When broad breadth is high (most
stocks participating in rally), QQQ is likely to continue its uptrend. When
breadth narrows below 50%, the rally is narrow and fragile.

This is fundamentally different from SP500 stock-picking:
  - HOLDS QQQ or IEF (ETFs), NOT individual SP500 stocks
  - Uses SP500 stocks for SIGNAL ONLY (breadth computation), not as holdings
  - Mechanically similar to RSP/SPY breadth ratio strategies but uses absolute
    fraction > 50d SMA threshold, not relative SPY comparison

Design:
  - Compute fraction of SP500 stocks (tradeable subset) above their 50d SMA.
  - High breadth (>= 60%): hold QQQ at 97% (strong bull confirmation)
  - Medium breadth (40-60%): hold SPY at 75% + IEF at 22%
  - Low breadth (< 40%): hold IEF at 97%
  - SPY 200d SMA outer bear gate: override all to IEF when SPY bearish
  - Rebalance every 5 bars (weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5         # weekly
BREADTH_SMA_WINDOW = 50     # per-stock SMA for breadth computation
SPY_TREND_WINDOW = 200      # outer trend gate
HIGH_BREADTH_THRESHOLD = 0.60
LOW_BREADTH_THRESHOLD = 0.40
EXPOSURE = 0.97


class SP500BreadthQQQTiming(Strategy):
    """SP500 breadth (% stocks > 50d SMA) gates QQQ/SPY/IEF allocation;
    high breadth = QQQ 97%; mid = SPY 75%+IEF 22%; low = IEF 97%;
    SPY 200d outer bear gate to IEF; weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_sma_window: int = BREADTH_SMA_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        high_breadth: float = HIGH_BREADTH_THRESHOLD,
        low_breadth: float = LOW_BREADTH_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_sma_window=breadth_sma_window,
            spy_trend_window=spy_trend_window,
            high_breadth=high_breadth,
            low_breadth=low_breadth,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_sma_window = int(breadth_sma_window)
        self.spy_trend_window = int(spy_trend_window)
        self.high_breadth = float(high_breadth)
        self.low_breadth = float(low_breadth)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.breadth_sma_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Compute SP500 breadth: fraction of stocks above 50d SMA
            need = self.breadth_sma_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                target["IEF"] = self.exposure
            else:
                above_count = 0
                total_count = 0

                for sym in prices.columns:
                    # Skip ETFs we're going to hold
                    if sym in ("QQQ", "SPY", "IEF", "SHY"):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.breadth_sma_window:
                        continue
                    arr = col.values
                    sma = float(np.mean(arr[-self.breadth_sma_window:]))
                    last = float(arr[-1])
                    total_count += 1
                    if last > sma:
                        above_count += 1

                if total_count < 50:
                    # Not enough stocks to compute reliable breadth
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    breadth = above_count / total_count

                    if breadth >= self.high_breadth:
                        # High breadth: QQQ bull
                        if "QQQ" in closes_now.index:
                            target["QQQ"] = self.exposure
                    elif breadth < self.low_breadth:
                        # Low breadth: defensive
                        if "IEF" in closes_now.index:
                            target["IEF"] = self.exposure
                    else:
                        # Medium breadth: blended
                        if "SPY" in closes_now.index:
                            target["SPY"] = 0.75 * self.exposure
                        if "IEF" in closes_now.index:
                            target["IEF"] = 0.22 * self.exposure

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["QQQ", "SPY", "IEF"]


NAME = "sp500_breadth_qqq_timing"
HYPOTHESIS = (
    "SP500 internal breadth (fraction of stocks above 50d SMA) gates QQQ/SPY/IEF allocation: "
    "high breadth (>=60%) hold QQQ 97%; mid breadth (40-60%) hold SPY 75%+IEF 22%; low breadth "
    "(<40%) hold IEF 97%; SPY 200d outer bear gate to IEF; weekly rebalance — holds ETFs "
    "(not individual stocks), uses SP500 breadth as regime signal, orthogonal to all stock-picking "
    "strategies on leaderboard"
)

UNIVERSE = _universe

STRATEGY = SP500BreadthQQQTiming()
