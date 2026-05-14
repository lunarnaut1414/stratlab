"""Multi-Timeframe Momentum Confirmation on SP500 — gen_8 sonnet-4

Hypothesis: Hold top-15 SP500 stocks where ALL THREE momentum lookbacks
(21d, 63d, 126d) are simultaneously positive AND the stock is above its
200d SMA. Equal-weight. SPY 200d SMA market gate to TLT. Biweekly rebalance.

Rationale: Most momentum strategies use a single window. Triple-window
confirmation — requiring short (21d), medium (63d), and long (126d) momentum
to all be positive — acts as a strong filter against stocks in corrective
phases, early-stage breakouts from downtrends, or deteriorating trends.
The composite score (sum of all three returns) ranks within confirmed candidates.

Distinction from existing strategies:
- All existing momentum strategies use a SINGLE momentum window (42d, 63d, 126d).
  This requires ALL THREE to be simultaneously positive — a much stricter filter.
- The triple-window AND condition produces a very different selection than any
  single window alone, especially at trend inflection points.
- Different from idiosyncratic momentum (which adjusts for beta, not multi-window).
- Different from 52w-high strategies (which use max-price proximity, not return sign).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # bi-weekly
WIN_SHORT = 21             # 1-month momentum
WIN_MED = 63               # 3-month momentum
WIN_LONG = 126             # 6-month momentum
STOCK_TREND = 200          # individual stock 200d SMA filter
TREND_WINDOW = 200         # SPY market-wide trend gate
TOP_K = 15
EXPOSURE = 0.97


class MultiwindowMomentumConfirm(Strategy):
    """SP500 triple-window momentum confirmation with SPY 200d gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        win_short: int = WIN_SHORT,
        win_med: int = WIN_MED,
        win_long: int = WIN_LONG,
        stock_trend: int = STOCK_TREND,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            win_short=win_short,
            win_med=win_med,
            win_long=win_long,
            stock_trend=stock_trend,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.win_short = int(win_short)
        self.win_med = int(win_med)
        self.win_long = int(win_long)
        self.stock_trend = int(stock_trend)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.win_long, self.trend_window, self.stock_trend) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA market gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT defensive
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Need enough history for all windows + stock trend
            need = max(self.win_long, self.stock_trend) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.win_long + 2:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                # Skip non-tradeable and auxiliary symbols
                if sym.startswith("^") or sym.endswith("=F") or sym.endswith("=X"):
                    continue
                if sym in ("SPY", "TLT"):
                    continue

                col = prices[sym].dropna()
                if len(col) < self.win_long + 2:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0:
                    continue

                # Individual stock 200d SMA filter
                if len(col) >= self.stock_trend:
                    sma200 = float(col.iloc[-self.stock_trend:].mean())
                    if current_price <= sma200:
                        continue

                # Compute all three momentum windows
                if len(col) < self.win_short + 1:
                    continue
                p_short = float(col.iloc[-self.win_short])
                if p_short <= 0:
                    continue
                ret_short = current_price / p_short - 1.0

                if len(col) < self.win_med + 1:
                    continue
                p_med = float(col.iloc[-self.win_med])
                if p_med <= 0:
                    continue
                ret_med = current_price / p_med - 1.0

                if len(col) < self.win_long + 1:
                    continue
                p_long = float(col.iloc[-self.win_long])
                if p_long <= 0:
                    continue
                ret_long = current_price / p_long - 1.0

                # Triple-confirmation: ALL three must be positive
                if not (np.isfinite(ret_short) and np.isfinite(ret_med) and np.isfinite(ret_long)):
                    continue
                if ret_short <= 0 or ret_med <= 0 or ret_long <= 0:
                    continue

                # Composite score = sum of all three returns (rank by total momentum)
                composite = ret_short + ret_med + ret_long
                scores[sym] = composite

            if len(scores) < 5:
                # Not enough triple-confirmed candidates — TLT defensive
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
                        target[sym] = per_weight

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "multiwindow_momentum_confirm"
HYPOTHESIS = (
    "Multi-timeframe momentum confirmation on SP500: hold top-15 SP500 stocks where all three "
    "momentum lookbacks (21d, 63d, 126d) are simultaneously positive AND stock is above 200d SMA; "
    "equal-weight; SPY 200d SMA market gate to TLT; biweekly rebalance; triple-window confirmation "
    "reduces false positives from single-window momentum"
)

UNIVERSE = _universe

STRATEGY = MultiwindowMomentumConfirm()
