"""SP500 126d Momentum with Individual 50d SMA + Golden Cross Gate — gen_7 sonnet-3

Hypothesis: Hold top-20 SP500 stocks by 126d return that are above their
own 50d SMA (individual trend confirmation); equal-weight; SPY 50d vs 150d MA
golden cross gate (faster than 200d SMA); IEF defensive; biweekly rebalance.

Rationale: Filtering stocks to those above their own 50d SMA (individual trend)
is different from SPY's market-wide trend check — it ensures each holding is
in its own short-term uptrend. The 50d/150d golden cross on SPY is faster
than the 200d SMA approach, catching trend changes earlier. IEF as defensive
(mid-duration) vs TLT reduces interest rate risk.

Distinction from existing strategies:
- Individual 50d SMA per stock (not just market-wide trend gate)
- 50d/150d golden cross on SPY (faster than standard 200d gate)
- IEF defensive (not TLT)
- 126d lookback with equal-weight (vs inverse-vol weighted skip-month variants)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bi-weekly
MOMENTUM_WINDOW = 126     # ~6 months
STOCK_SMA = 50            # individual stock trend window
FAST_MA = 50              # SPY fast MA for golden cross
SLOW_MA = 150             # SPY slow MA for golden cross
TOP_K = 20
EXPOSURE = 0.97


class Sp500_126dStock50SmaGoldencross(Strategy):
    """SP500 126d momentum with individual 50d SMA filter and SPY 50/150 golden cross."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_sma: int = STOCK_SMA,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stock_sma=stock_sma,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stock_sma = int(stock_sma)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 50d/150d golden cross gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.slow_ma + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.slow_ma:
            return []

        spy_fast_sma = float(spy_close.iloc[-self.fast_ma:].mean())
        spy_slow_sma = float(spy_close.iloc[-self.slow_ma:].mean())
        bull = spy_fast_sma > spy_slow_sma  # golden cross condition

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Need enough history for momentum + stock SMA
            need = max(self.momentum_window, self.stock_sma) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                current_price = float(col.iloc[-1])

                # Individual stock 50d SMA filter
                if len(col) >= self.stock_sma:
                    stock_sma = float(col.iloc[-self.stock_sma:].mean())
                    if current_price <= stock_sma:
                        continue  # Stock below its own 50d SMA

                # 126d momentum
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(current_price):
                    continue
                ret = current_price / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                scores[sym] = ret

            if len(scores) < 5:
                # Not enough candidates — IEF defensive
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = weight

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
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_126d_stock_50sma_goldencross"
HYPOTHESIS = (
    "SP500 stocks above individual 50d SMA with top-20 126d momentum: "
    "hold top-20 SP500 stocks by 126d return that are above their own 50d SMA; "
    "equal-weight; SPY 50d vs 150d MA golden cross gate; IEF defensive; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = Sp500_126dStock50SmaGoldencross()
