"""SP500 low-drawdown momentum — gen_6 sonnet-7

Hypothesis: Hold top-20 SP500 stocks ranked by 63d momentum, filtered
to stocks whose 63d maximum intra-period drawdown is less than -15%
(quality filter — excludes momentum stocks with volatile/choppy paths).
Inverse-vol weighted. SPY 200d SMA gate; TLT defensive. Biweekly rebalance.

Rationale:
  Standard momentum portfolios include high-vol "lottery" names that
  spike briefly then collapse. Adding a drawdown quality filter retains
  momentum names that trend smoothly rather than spiking — these tend
  to have lower crash risk and more sustained momentum. The inverse-vol
  weighting further stabilizes the portfolio.

  Distinct from existing leaderboard:
  - 63d max-drawdown quality filter (not in any existing strategy)
  - Inverse-vol weighting on top of drawdown filter
  - Combined quality+momentum composite without 52wk-high proximity
  - SPY 200d SMA gate (no VIX or credit signal)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 63     # ~3 months
MAX_DD_THRESHOLD = -0.15  # max allowed 63d drawdown (-15%)
VOL_WINDOW = 20          # for inverse-vol weighting
TOP_K = 20
TREND_WINDOW = 200       # SPY 200d SMA gate
EXPOSURE = 0.97


class SP500LowDDMomentum(Strategy):
    """SP500 momentum with 63d max-drawdown quality filter + inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        max_dd_threshold: float = MAX_DD_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            max_dd_threshold=max_dd_threshold,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.max_dd_threshold = float(max_dd_threshold)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + self.vol_window + 5
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
            # Bear market: hold TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Bull market: momentum + drawdown quality filter
            need = self.momentum_window + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.vol_window:
                    continue
                current = float(col.iloc[-1])
                start = float(col.iloc[-self.momentum_window])
                if start <= 0 or not np.isfinite(start) or not np.isfinite(current):
                    continue

                # Momentum
                ret = current / start - 1.0
                if not np.isfinite(ret):
                    continue

                # 63d max drawdown quality filter
                window_prices = col.iloc[-self.momentum_window:]
                rolling_max = window_prices.cummax()
                drawdowns = (window_prices - rolling_max) / rolling_max
                max_dd = float(drawdowns.min())  # most negative = worst drawdown

                # Reject stocks with too steep a drawdown in the period
                if max_dd < self.max_dd_threshold:
                    continue

                # Inverse vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Fall back to TLT if too few qualifying stocks
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_lowdd_momentum"
HYPOTHESIS = (
    "SP500 momentum with 63d max-drawdown quality filter: hold top-20 SP500 stocks by 63d "
    "momentum filtered to stocks with 63d max-drawdown > -15% (smooth trends only), "
    "inverse-vol weighted; SPY 200d SMA gate; TLT defensive; biweekly rebalance"
)
UNIVERSE = _universe
STRATEGY = SP500LowDDMomentum()
