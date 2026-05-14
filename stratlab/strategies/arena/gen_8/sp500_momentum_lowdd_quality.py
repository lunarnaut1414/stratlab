"""SP500 Momentum-Quality via Low-Drawdown Filter — gen_8 sonnet-2

Hypothesis: Rank SP500 stocks by a composite score of 126d return * (1 - 63d max
drawdown), selecting stocks with strong sustained momentum but low peak-to-trough
volatility. This quality filter avoids "lottery momentum" stocks that spike then
collapse. Hold top-15 equal-weight when SPY above 200d SMA. TLT defensive.
Biweekly rebalance.

Rationale:
- Pure momentum often selects high-beta stocks that draw down severely on reversals
- Penalizing max-drawdown in the score filters for smooth, persistent momentum
  (stocks rising steadily vs. spike-and-crash patterns)
- Distinct from idiosyncratic_momentum (beta-adjusted), nearhi_quality (52w high
  proximity), and low-beta (ascending beta sort)
- The composite score selects for quality of trend, not just magnitude

Composite = 126d_return * (1 - 63d_max_drawdown)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # 6 months
DD_WINDOW = 63            # 3 months for drawdown calculation
TREND_WINDOW = 200        # SPY 200d SMA
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"


class SP500MomentumLowDD(Strategy):
    """SP500 momentum with low-drawdown quality filter; TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        dd_window: int = DD_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            dd_window=dd_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.dd_window = int(dd_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
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
            # Bear regime: defensive TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _TLT):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 1:
                    continue

                # 126d raw return
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                raw_ret = p_end / p_start - 1.0
                if not np.isfinite(raw_ret):
                    continue

                # 63d max drawdown (worst peak-to-trough over the window)
                if len(col) >= self.dd_window + 1:
                    dd_slice = col.iloc[-self.dd_window - 1:]
                else:
                    dd_slice = col
                vals = dd_slice.values
                if len(vals) < 2:
                    continue
                running_max = np.maximum.accumulate(vals)
                dd_series = vals / running_max - 1.0
                max_dd = float(np.min(dd_series))  # most negative = worst drawdown
                # Clamp to [-1, 0]
                max_dd = max(-1.0, min(0.0, max_dd))

                # Composite: reward high momentum * low drawdown
                # (1 - 63d_max_dd_magnitude) boosts stocks with small pullbacks
                quality_adj = 1.0 - abs(max_dd)  # in [0, 1], higher = smaller drawdown
                composite = raw_ret * quality_adj
                if np.isfinite(composite):
                    scores[sym] = composite

            if len(scores) < 5:
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_wt = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
                        target[sym] = per_wt

        orders: list[Order] = []

        # Exit positions not in target
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
    return sp500_tickers() + [_TLT, _SPY]


NAME = "sp500_momentum_lowdd_quality"
HYPOTHESIS = (
    "SP500 momentum-quality: rank SP500 stocks by composite score of 126d return "
    "* (1 - 63d max drawdown magnitude); hold top-15 equal-weight when SPY > 200d "
    "SMA; TLT defensive in bear; biweekly rebalance; selects stocks with smooth "
    "sustained trends vs spike-and-crash momentum"
)

UNIVERSE = _universe

STRATEGY = SP500MomentumLowDD()
