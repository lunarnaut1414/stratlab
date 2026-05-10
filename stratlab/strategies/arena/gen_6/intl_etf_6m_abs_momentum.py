"""International ETF absolute momentum rotation (6m) — gen_6 sonnet-7

Hypothesis: Rank EWJ/EWG/EWU/EWC/EWA/EEM/EFA by 126-day (6-month) momentum;
hold top-3 only when their 126d return is positive (absolute momentum gate);
hold TLT when fewer than 2 qualify; SPY 200d SMA additional bear-market gate
to exit all international equity. Biweekly rebalance.

Distinct from intl_country_etf_rotation.py (sonnet-4):
  - Uses 6-month (126d) lookback vs 3-month (63d)
  - Absolute momentum gate: only hold countries with positive 6m return
  - TLT as defensive vs IEF+TLT split
  - Dual gate: both absolute momentum AND SPY trend

Rationale: International momentum at 6-month horizon is better supported
by academic literature (Fama-French international momentum at 6-12 months).
The absolute momentum filter prevents holding weak international markets
that are in a bear regime. Combining SPY trend gate with per-country
absolute momentum is more selective than pure relative ranking.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

INTL_ETFS = ["EWJ", "EWU", "EWC", "EWA", "EEM", "EFA", "EWY", "EWZ", "EWT"]

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 126    # 6 months
TREND_WINDOW = 200       # SPY 200d SMA
TOP_K = 3
MIN_QUALIFY = 2          # need at least 2 positive-momentum ETFs to go risk-on
EXPOSURE = 0.97


class IntlETF6mAbsMomentum(Strategy):
    """International ETF rotation: 6m absolute momentum with SPY trend gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        min_qualify: int = MIN_QUALIFY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            min_qualify=min_qualify,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.min_qualify = int(min_qualify)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- Trend filter: SPY 200d SMA ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
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
            # Global bear: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Compute 6-month momentum for each country ETF
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in INTL_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            # Absolute momentum filter: only countries with positive 6m return
            positive = {s: r for s, r in scores.items() if r > 0}

            if len(positive) < self.min_qualify:
                # Fewer than min_qualify countries have positive momentum: hold TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Hold top-K among positive-momentum countries
                k = min(self.top_k, len(positive))
                ranked = sorted(positive, key=positive.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        # Build orders
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


NAME = "intl_etf_6m_abs_momentum"
HYPOTHESIS = (
    "International ETF 6-month absolute momentum: rank EWJ/EWG/EWU/EWC/EWA/EEM/EFA by 126d "
    "return, hold top-3 only when at least 2 have positive 6m return; TLT when fewer qualify; "
    "SPY 200d SMA bear-market gate; biweekly rebalance; 6-month horizon + absolute filter"
)
UNIVERSE = ["EWJ", "EWU", "EWC", "EWA", "EEM", "EFA", "EWY", "EWZ", "EWT", "TLT", "SPY"]
STRATEGY = IntlETF6mAbsMomentum()
