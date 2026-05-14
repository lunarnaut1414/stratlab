"""SP500 Multi-Trend Quality Momentum — gen_8 sonnet-10

Hypothesis: Score each SP500 stock by how many of its 50d/100d/200d SMAs it
is above (0-3 quality points). Among stocks with full quality score (above all 3
SMAs simultaneously), rank by 63d momentum and hold top-15. When fewer than 15
stocks achieve full trend alignment, relax to 2+ trend points to fill remaining
slots.

Rationale: Being above the 200d SMA is a necessary but not sufficient condition
for trend quality. A stock above ALL of 50d, 100d, 200d is in confirmed uptrend
across short, medium, and long time horizons — these are the strongest trending
names. This multi-SMA filter is substantially stricter than the 200d-only gate
used by most leaderboard strategies, producing a more concentrated and higher-
conviction portfolio.

SPY 200d SMA gate: avoid stock selection entirely in bear regimes.
IEF defensive (different from TLT to reduce overlap with defensive bucket).
Monthly rebalance: 21 bars to reduce turnover vs biweekly variants.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21      # monthly
MOMENTUM_WINDOW = 63      # ~3 months
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"

# Multi-timeframe SMA windows
_SHORT_SMA = 50
_MED_SMA = 100
_LONG_SMA = 200


class SP500MultiTrendQualityMomentum(Strategy):
    """SP500 momentum filtered by multi-SMA trend quality (0-3 points)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        short_sma: int = _SHORT_SMA,
        med_sma: int = _MED_SMA,
        long_sma: int = _LONG_SMA,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            top_k=top_k,
            exposure=exposure,
            short_sma=short_sma,
            med_sma=med_sma,
            long_sma=long_sma,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.short_sma = int(short_sma)
        self.med_sma = int(med_sma)
        self.long_sma = int(long_sma)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.long_sma + self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA bear gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.long_sma + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.long_sma:
            return []
        spy_sma_200 = float(spy_close.iloc[-self.long_sma:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma_200

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            need = self.long_sma + self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                return []

            # Score each stock: trend quality + momentum
            quality_3: list[tuple[str, float]] = []  # above all 3 SMAs
            quality_2: list[tuple[str, float]] = []  # above 2 SMAs (med + long)

            for sym in prices.columns:
                if sym in (_SPY, _IEF):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.long_sma + 2:
                    continue

                price_now = float(col.iloc[-1])
                if price_now <= 0:
                    continue

                # Compute SMAs
                sma_short = float(col.iloc[-self.short_sma:].mean()) if len(col) >= self.short_sma else None
                sma_med = float(col.iloc[-self.med_sma:].mean()) if len(col) >= self.med_sma else None
                sma_long = float(col.iloc[-self.long_sma:].mean()) if len(col) >= self.long_sma else None

                above_long = sma_long is not None and price_now > sma_long
                above_med = sma_med is not None and price_now > sma_med
                above_short = sma_short is not None and price_now > sma_short

                # Must be above at least the 200d SMA (long-term trend)
                if not above_long:
                    continue

                # 63d momentum
                if len(col) < self.momentum_window + 1:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(ret):
                    continue

                # Quality classification
                quality_score = sum([above_long, above_med, above_short])
                if quality_score == 3:
                    quality_3.append((sym, ret))
                elif quality_score == 2 and above_long and above_med:
                    quality_2.append((sym, ret))

            # Sort by momentum within quality tier
            quality_3.sort(key=lambda x: x[1], reverse=True)
            quality_2.sort(key=lambda x: x[1], reverse=True)

            # Fill top_k from tier 3 first, then tier 2
            selected: list[str] = []
            for sym, _ in quality_3:
                if len(selected) >= self.top_k:
                    break
                if sym in live:
                    selected.append(sym)

            # Fill remaining slots from tier 2 if needed
            if len(selected) < 5:
                for sym, _ in quality_2:
                    if len(selected) >= self.top_k:
                        break
                    if sym not in selected and sym in live:
                        selected.append(sym)

            if not selected:
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                per_weight = self.exposure / len(selected)
                for sym in selected:
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
    return sp500_tickers() + [_IEF, _SPY]


NAME = "sp500_multi_trend_quality"
HYPOTHESIS = (
    "SP500 multi-trend quality momentum: score each SP500 stock by how many of its "
    "50d/100d/200d SMAs it is above (0-3 trend quality points), then rank by 63d momentum "
    "among stocks with quality score of 3; hold top-15; SPY 200d SMA gate; IEF defensive; "
    "monthly rebalance; multi-SMA filter is stricter than single 200d gate and selects only "
    "stocks in confirmed trend across timeframes"
)

UNIVERSE = _universe

STRATEGY = SP500MultiTrendQualityMomentum()
