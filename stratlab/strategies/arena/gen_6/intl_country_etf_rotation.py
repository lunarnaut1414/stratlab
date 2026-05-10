"""International Country ETF Rotation — gen_6 sonnet-4

Hypothesis: Rotate among international country and regional ETFs based on
63-day momentum. Hold top-3 ETFs in equity expansion (SPY above 200d SMA),
and rotate to IEF+TLT 50/50 when SPY in bear territory.

Universe: EWJ (Japan), EWG (Germany), EWU (UK), EWC (Canada), EWA (Australia),
EEM (Emerging Markets), EFA (Developed Intl), SPY (as comparison anchor)

Signal: 63d price return for each country ETF
Gate: SPY 200d SMA for bull/bear regime
Rebalance: every 10 bars (biweekly)

Distinct from leaderboard:
  - Cross-border geographic diversification (untouched in gen_5)
  - No VIX, no credit spread, no yield curve signal
  - Captures international equity risk premium with simple trend rotation
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
TOP_K = 3
EXPOSURE = 0.97

COUNTRY_ETFS = ["EWJ", "EWG", "EWU", "EWC", "EWA", "EEM", "EFA"]
DEFENSIVE_ETFS = ["IEF", "TLT"]


class IntlCountryETFRotation(Strategy):
    """International country ETF rotation gated by SPY 200d SMA."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: IEF 50% + TLT 50%
            avail_def = [s for s in DEFENSIVE_ETFS if s in closes_now.index]
            if avail_def:
                per_weight = self.exposure / len(avail_def)
                for sym in avail_def:
                    target[sym] = per_weight
        else:
            # Risk-on: rank country ETFs by 63d momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in COUNTRY_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_now = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0:
                    continue
                ret = p_now / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                return []

            # Take top-K with positive momentum
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            selected = [s for s in ranked[:self.top_k] if scores[s] > 0]

            if not selected:
                # All negative momentum, go defensive
                avail_def = [s for s in DEFENSIVE_ETFS if s in closes_now.index]
                if avail_def:
                    per_weight = self.exposure / len(avail_def)
                    for sym in avail_def:
                        target[sym] = per_weight
            else:
                per_weight = self.exposure / len(selected)
                for sym in selected:
                    target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "intl_country_etf_rotation"
HYPOTHESIS = (
    "International country ETF rotation: rank EWJ/EWG/EWU/EWC/EWA/EEM/EFA by 63d momentum, "
    "hold top-3 equally when SPY above 200d SMA; hold IEF+TLT 50/50 when SPY below 200d SMA; "
    "biweekly rebalance; cross-border equity diversification untouched by gen_5 leaderboard"
)

UNIVERSE = ["EWJ", "EWG", "EWU", "EWC", "EWA", "EEM", "EFA", "IEF", "TLT", "SPY"]

STRATEGY = IntlCountryETFRotation()
