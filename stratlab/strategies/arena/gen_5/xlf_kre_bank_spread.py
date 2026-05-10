"""SP500 Low-Volatility Factor with Bull-Market Gate — gen_5 sonnet-9

Hypothesis:
  Hold the top-15 SP500 stocks with the LOWEST 20-day realized volatility
  (low-vol factor) when SPY is above its 200-day SMA (bull market confirmed).
  When SPY breaks below 200d SMA, rotate to TLT (bonds).

  Rationale: The low-volatility anomaly is well-documented — low-vol stocks
  have historically delivered higher risk-adjusted returns than high-vol stocks.
  This is distinct from momentum (which buys recent winners) and should have
  low correlation to the momentum-heavy strategies already on the leaderboard.

  Additional constraint: each stock must also be above its own 50-day SMA to
  filter out stocks that are low-vol because they're in a "quiet decline" phase.
  Rebalance every 10 bars (biweekly) to capture factor persistence while
  generating sufficient trade count.

IS window: 2010-01-01 to 2018-12-31
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Use SP500 universe
UNIVERSE = "sp500"

VOL_WINDOW = 20       # 20-day realized vol window for ranking
TREND_WINDOW = 200    # 200d SMA for SPY regime filter
STOCK_TREND = 50      # 50d SMA for individual stock filter
TOP_K = 15            # hold top-15 low-vol stocks
REBALANCE_DAYS = 10   # biweekly rebalance
MIN_HISTORY = max(VOL_WINDOW, TREND_WINDOW, STOCK_TREND) + 5
EXPOSURE = 0.97


class LowVolSp500Strategy(Strategy):
    """SP500 low-volatility factor with 200d SMA bull-market gate."""

    def __init__(
        self,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        stock_trend: int = STOCK_TREND,
        top_k: int = TOP_K,
        rebalance_days: int = REBALANCE_DAYS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            vol_window=vol_window,
            trend_window=trend_window,
            stock_trend=stock_trend,
            top_k=top_k,
            rebalance_days=rebalance_days,
            exposure=exposure,
        )
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.stock_trend = stock_trend
        self.top_k = top_k
        self.rebalance_days = rebalance_days
        self.exposure = exposure
        self._bar_count: int = 0

    def on_start(self) -> None:
        self._bar_count = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        if ctx.idx < MIN_HISTORY:
            return []

        self._bar_count += 1
        if self._bar_count % self.rebalance_days != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        # SPY trend filter: need SPY in our universe
        live_closes = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # Use closes_window for cross-sectional computation
        needed = max(self.trend_window, self.stock_trend, self.vol_window) + 5
        prices = ctx.closes_window(needed)
        if len(prices) < self.vol_window + 2:
            return []

        # Check SPY trend
        spy_above_trend = True
        if "SPY" in prices.columns:
            spy_col = prices["SPY"].dropna()
            if len(spy_col) >= self.trend_window + 1:
                spy_current = float(spy_col.iloc[-1])
                spy_ma = float(spy_col.iloc[-self.trend_window:].mean())
                if np.isfinite(spy_current) and np.isfinite(spy_ma):
                    spy_above_trend = spy_current > spy_ma

        equity = ctx.portfolio_value(live_closes)
        if equity <= 0:
            return []

        if not spy_above_trend:
            # Bear market: hold TLT
            target: dict[str, int] = {}
            if "TLT" in ctx.symbols and live_closes.get("TLT", 0) > 0:
                price = live_closes["TLT"]
                shares = int(equity * self.exposure / price)
                if shares > 0:
                    target["TLT"] = shares
        else:
            # Bull market: rank by realized vol (ascending = low vol is best)
            vol_scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym == "SPY" or sym == "TLT":
                    continue
                col = prices[sym].dropna()
                if len(col) < self.vol_window + 2:
                    continue

                # Individual stock trend filter: above 50d SMA
                if len(col) >= self.stock_trend + 1:
                    stock_ma = float(col.iloc[-self.stock_trend:].mean())
                    stock_price = float(col.iloc[-1])
                    if np.isfinite(stock_ma) and np.isfinite(stock_price):
                        if stock_price < stock_ma:
                            continue  # skip stocks in downtrend

                # Compute 20-day annualized volatility
                returns = col.iloc[-self.vol_window:].pct_change().dropna()
                if len(returns) < self.vol_window // 2:
                    continue
                ann_vol = float(returns.std() * np.sqrt(252))
                if np.isfinite(ann_vol) and ann_vol > 0:
                    vol_scores[sym] = ann_vol

            if not vol_scores:
                # Fallback: hold SPY
                target = {}
                if "SPY" in ctx.symbols and live_closes.get("SPY", 0) > 0:
                    price = live_closes["SPY"]
                    shares = int(equity * self.exposure / price)
                    if shares > 0:
                        target["SPY"] = shares
            else:
                # Sort ascending by vol (lowest vol first)
                ranked = sorted(vol_scores, key=vol_scores.__getitem__)
                longs = [s for s in ranked[:self.top_k] if s in live_closes and live_closes[s] > 0]

                if not longs:
                    target = {}
                else:
                    per_weight = self.exposure / len(longs)
                    target = {}
                    for sym in longs:
                        price = live_closes[sym]
                        shares = int(equity * per_weight / price)
                        if shares > 0:
                            target[sym] = shares

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, tgt_shares in target.items():
            current = int(ctx.position(sym).size)
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "TLT"]


NAME = "xlf_kre_bank_spread"
HYPOTHESIS = (
    "SP500 low-volatility factor with 200d SMA bull-market gate: hold top-15 SP500 stocks "
    "by lowest 20-day realized vol (above their 50d SMA) in bull market (SPY>200d); "
    "rotate to TLT in bear market; biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = LowVolSp500Strategy()
