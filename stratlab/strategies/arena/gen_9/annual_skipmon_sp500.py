"""gen_9 sonnet-6 — Annual Skip-Month Momentum on SP500

Hypothesis: Rank SP500 stocks by 12-month-minus-1-month return (252d return
minus the most recent 21d return, i.e. the return from 252 bars ago to 21 bars
ago). This classic Jegadeesh-Titman "annual skip-month" avoids short-term
reversal contamination while capturing the full-year momentum factor.

Hold top-15 above 200d SMA, equal-weight. Monthly rebalance (21 bars).
SPY 200d SMA outer bear gate → TLT.

Rationale: The 252d-21d skip-month is the LONGEST lookback momentum in the
literature (vs 126d-21d in gen_8, 63d in most gen_5-7 strategies). Annual
winners tend to be structural outperformers — businesses with sustained
competitive advantages that have compounded returns over a full year. The
skip on the most recent month avoids buying into a short-term spike. Monthly
rebalance (not biweekly) makes this a slower-turnover strategy that captures
a different segment of the momentum premium.

IS Calmar should be strong given the long lookback captures 2010-2018's secular
winners (tech, healthcare, consumer discretionary multi-year compounders).

Note: Does NOT use any additional macro gate — SPY 200d trend is the only
filter. Simpler than multi-signal approaches; the annual momentum signal carries
the edge.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21   # monthly
MOM_LONG = 252         # 12-month lookback
MOM_SKIP = 21          # skip most recent 1 month
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"


class AnnualSkipmonSP500(Strategy):
    """SP500 annual skip-month momentum with SPY 200d SMA gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.mom_long) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue

                    # Annual skip-month: return from 252 bars ago to 21 bars ago
                    p_long_ago = float(col.iloc[-self.mom_long])
                    p_skip_ago = float(col.iloc[-self.mom_skip])
                    if p_long_ago <= 0:
                        continue
                    skip_mom = p_skip_ago / p_long_ago - 1.0
                    if np.isfinite(skip_mom):
                        scores[sym] = skip_mom

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY]


NAME = "annual_skipmon_sp500"
HYPOTHESIS = (
    "Annual momentum (252d-21d skip-month) on SP500 stocks with SPY trend gate: rank SP500 "
    "stocks by 252d return minus 21d return (annual skip-month, skipping recency reversal), "
    "hold top-15 above 200d SMA, equal-weight; SPY 200d outer bear gate to TLT; monthly "
    "rebalance (21 bars) for stability; distinct from 126d-21d and 63d lookbacks"
)

UNIVERSE = _universe

STRATEGY = AnnualSkipmonSP500()
