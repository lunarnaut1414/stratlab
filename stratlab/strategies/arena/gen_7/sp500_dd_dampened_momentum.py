"""SP500 drawdown-dampened momentum strategy.

Hypothesis: buy top-20 SP500 stocks ranked by composite score =
63d momentum * (1 - 63d max drawdown), inverse-vol weighted;
SPY 200d SMA gate; TLT defensive; rebalance every 10 bars.

Rationale: Pure momentum strategies buy the highest-returning stocks, but some
of those gains come from volatile, erratic price paths that are harder to hold
and prone to sharp reversals. By dampening the momentum score with the
max-drawdown over the same window, we prefer stocks that climbed steadily (high
return, low drawdown) over lottery-like climbers (high return, large intra-period
drawdown). The result is a quality-tilted momentum factor distinct from simple
return ranking, near-high proximity, or Sharpe ranking.

Key distinctions from leaderboard:
  - Composite score = momentum * (1 - max_dd) not seen on leaderboard
  - Different from nearhi_momentum_quality (uses near-high ratio, not drawdown)
  - Different from sp500_lowvol_factor (sorts by vol not composite score)
  - 10-bar rebalance generates high trade count
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars
MOMENTUM_WINDOW = 63      # ~3 months
VOL_WINDOW = 20           # for inverse-vol weights
TOP_K = 20
TREND_WINDOW = 200
EXPOSURE = 0.97


class SP500DrawdownDampenedMomentum(Strategy):
    """Drawdown-dampened momentum: composite = 63d_return * (1 - 63d_maxdd),
    inverse-vol weighted; SPY 200d SMA gate; TLT defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA regime gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
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
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
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

                # 63d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret) or ret <= 0:
                    continue

                # 63d max drawdown (from peak over window)
                window_prices = col.iloc[-self.momentum_window:]
                rolling_max = window_prices.expanding().max()
                drawdowns = (window_prices - rolling_max) / rolling_max
                max_dd = float(abs(drawdowns.min()))  # positive number 0..1

                # Composite: dampen momentum by drawdown severity
                # A stock with ret=0.20 and max_dd=0.05 scores 0.20 * 0.95 = 0.19
                # A stock with ret=0.20 and max_dd=0.15 scores 0.20 * 0.85 = 0.17
                composite = ret * (1.0 - max_dd)
                if not np.isfinite(composite):
                    continue

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = composite
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_dd_dampened_momentum"
HYPOTHESIS = (
    "SP500 earnings-momentum composite: buy top-20 SP500 stocks ranked by 63d momentum * "
    "(1 - 63d max drawdown) composite score, inverse-vol weighted; SPY 200d SMA gate; "
    "TLT defensive; rebalance every 10 bars — drawdown-dampened momentum distinct from pure return ranking"
)

UNIVERSE = _universe

STRATEGY = SP500DrawdownDampenedMomentum()
