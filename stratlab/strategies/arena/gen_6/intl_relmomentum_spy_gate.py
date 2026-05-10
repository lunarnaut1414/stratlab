"""International ETF relative-momentum + SPY gate — gen_6 sonnet-7

Hypothesis: Rank EWJ/EWU/EWC/EWA/EEM/EFA by 63-day relative momentum
vs SPY (excess return over SPY over the same period). Hold top-2 countries
that beat SPY over 63 days when SPY is above 200d SMA. Hold SPY when SPY
is bullish but no international ETF beats it. Hold TLT when SPY is below
200d SMA. Rebalance biweekly (10 bars).

Rationale:
  International ETFs that outperform SPY over 63 days show genuine relative
  momentum versus the US market. This is different from absolute return
  ranking (which favors all ETFs in a US bull market). By requiring countries
  to BEAT SPY, the strategy allocates internationally only when there's true
  relative strength vs the US market. This reduces the correlation to US
  equity strategies since it rotates to SPY when no international market leads.

  Distinct from gen6_intl_country_etf_rotation (sonnet-4):
  - Uses relative momentum vs SPY (excess return) not absolute return
  - Holds top-2 (not top-3) to concentrate in stronger signals
  - Defaults to SPY (not defensive bonds) when no international market leads
  - TLT only as bear market protection (not as neutral-zone allocation)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

INTL_ETFS = ["EWJ", "EWU", "EWC", "EWA", "EEM", "EFA", "EWY", "EWZ"]
REBALANCE_EVERY = 10   # biweekly
MOMENTUM_WINDOW = 63   # ~3 months
TREND_WINDOW = 200     # SPY 200d SMA
TOP_K = 2              # hold top-2 (more concentrated)
EXPOSURE = 0.97


class IntlRelMomentumSPYGate(Strategy):
    """International ETF rotation using relative momentum vs SPY."""

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

        # SPY trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # SPY 63d return (benchmark)
        if len(spy_close) < self.momentum_window:
            return []
        spy_mom = float(spy_close.iloc[-1] / spy_close.iloc[-self.momentum_window] - 1.0)

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Compute relative momentum vs SPY for each international ETF
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            excess_returns: dict[str, float] = {}
            for sym in INTL_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    excess = ret - spy_mom  # relative to SPY
                    excess_returns[sym] = excess

            # Hold top-K countries that BEAT SPY (positive excess return)
            positive_excess = {s: r for s, r in excess_returns.items() if r > 0}

            if not positive_excess:
                # No international ETF beats SPY: hold SPY
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                k = min(self.top_k, len(positive_excess))
                ranked = sorted(positive_excess, key=positive_excess.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
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


NAME = "intl_relmomentum_spy_gate"
HYPOTHESIS = (
    "International ETF relative-momentum vs SPY: hold top-2 of EWJ/EWU/EWC/EWA/EEM/EFA/EWY/EWZ "
    "by excess 63d return over SPY when beating SPY; hold SPY when no country beats US; "
    "TLT when SPY<200d SMA; biweekly rebalance; relative not absolute international momentum"
)
UNIVERSE = INTL_ETFS + ["SPY", "TLT"]
STRATEGY = IntlRelMomentumSPYGate()
