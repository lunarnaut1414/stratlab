"""International country ETF momentum rotation strategy.

Hypothesis: Rank EWJ/EWG/EWU/EWC/EWA/EEM/EFA by 63d momentum and hold top-3
equally when SPY is above its 200d SMA (global risk-on); rotate to IEF 50% +
TLT 50% when SPY is below 200d SMA (global risk-off). Biweekly rebalance.

Rationale:
  Country ETFs are driven by local macro cycles and currency effects that are
  largely uncorrelated with individual SP500 stock selection. When global
  risk appetite is high (SPY trend up), the top-performing country markets
  capture international growth cycles. When SPY breaks the 200d SMA (global
  stress), flight to quality via treasury duration.

Diversification vs leaderboard:
  - All gen_5 accepted strategies trade US equities or ETFs (SPY/QQQ/sector).
  - International country ETFs (EWJ, EWG, EWU, EWC, EWA, EEM, EFA) are not
    in any existing leaderboard strategy.
  - This strategy derives returns from currency, local-equity, and
    cross-country growth differentials — very different return drivers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Country / broad international ETFs
INTL_ETFS = [
    "EWJ",   # Japan
    "EWG",   # Germany
    "EWU",   # United Kingdom
    "EWC",   # Canada
    "EWA",   # Australia
    "EEM",   # Emerging Markets
    "EFA",   # EAFE (broad developed ex-US)
]

MOMENTUM_WINDOW = 63   # ~3 months
REBALANCE_EVERY = 10   # bars (biweekly)
TOP_K = 3              # hold top 3 country ETFs
TREND_WINDOW = 200     # SPY 200d SMA gate
EXPOSURE = 0.97


class IntlETFRotation(Strategy):
    """International country ETF momentum rotation with SPY trend gate."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA trend gate ---
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
            # Risk-off: IEF 50% + TLT 50%
            for sym, wt in [("IEF", 0.5), ("TLT", 0.5)]:
                if sym in closes_now.index:
                    target[sym] = wt * self.exposure
        else:
            # Rank international ETFs by 63d momentum
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

            if len(scores) < 2:
                # Fallback to IEF if too few international ETFs available
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

        # Build orders: liquidate positions not in target
        orders: list[Order] = []
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


NAME = "intl_etf_rotation"
HYPOTHESIS = (
    "International country ETF rotation: rank EWJ/EWG/EWU/EWC/EWA/EEM/EFA by 63d momentum, "
    "hold top-3 equally when SPY above 200d SMA; hold IEF+TLT 50/50 when below; "
    "biweekly rebalance; cross-border equity diversification distinct from all gen_5 leaderboard strategies."
)

UNIVERSE = INTL_ETFS + ["IEF", "TLT", "SPY"]

STRATEGY = IntlETFRotation()
