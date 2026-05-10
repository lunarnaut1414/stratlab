"""SP500 short-term reversal — gen_6 sonnet-7

Hypothesis: Buy the 25 worst-performing SP500 stocks over the last 5
trading days (short-term reversal) when SPY is above its 200d SMA.
Equal-weight hold for 5 days then re-rank. When SPY is below 200d SMA,
rotate to TLT. Rebalance every 5 bars.

Rationale:
  Short-term stock return reversal (1-week or 1-month) is a well-documented
  anomaly — stocks that fall sharply over 1-5 days tend to bounce as
  market makers and arbitrageurs buy the dip. This is structurally the
  OPPOSITE of 1-3 month momentum strategies, so it should have low
  correlation to the momentum-heavy leaderboard strategies. The SPY 200d
  SMA gate prevents buying "dippers" in structural downtrends where
  short-term reversals fail.

  Distinct from existing leaderboard:
  - 5-day reversal (not momentum) as ranking signal
  - Buys worst performers (not best) — anti-momentum
  - Very different daily return path from all existing strategies
  - SPY 200d SMA gate prevents holding in bear markets
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5    # weekly
REVERSAL_WINDOW = 5    # 1-week return for ranking
TOP_K = 25             # 25 worst performers to buy
TREND_WINDOW = 200     # SPY 200d SMA gate
EXPOSURE = 0.97


class SP500ShortReversal(Strategy):
    """SP500 short-term reversal: buy worst 5d performers in a bull market."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        reversal_window: int = REVERSAL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            reversal_window=reversal_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.reversal_window = int(reversal_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.reversal_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Rank stocks by 5d return (ascending = worst = buy)
            prices = ctx.closes_window(self.reversal_window + 5)
            if len(prices) < self.reversal_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.reversal_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.reversal_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                return []

            # Buy the worst performers (short-term reversal)
            ranked = sorted(scores, key=scores.__getitem__)  # ascending = worst first
            buys = ranked[:self.top_k]
            per_weight = self.exposure / len(buys)
            for sym in buys:
                target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


NAME = "sp500_short_reversal"
HYPOTHESIS = (
    "SP500 5-day short-term reversal: buy 25 worst SP500 stocks by 5d return when SPY>200d SMA; "
    "TLT defensive when SPY<200d SMA; equal-weight 5-bar hold then re-rank; weekly rebalance; "
    "anti-momentum signal orthogonal to all existing momentum leaderboard strategies"
)
UNIVERSE = _universe
STRATEGY = SP500ShortReversal()
