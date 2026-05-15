"""SP500 Near-52w-High + Portfolio Vol-Targeting Momentum — gen_10 sonnet-6

Hypothesis: Combine the near-52w-high quality filter (price must be >90% of
252d high) with portfolio-level vol-targeting for regime-invariant deleveraging.
Rank filtered stocks by 126d momentum; hold top-15 above SPY 200d SMA;
inverse-vol weighted; IEF defensive.

Rationale:
  - Near-52w-high proximity acts as a quality filter: stocks near their yearly
    high show persistent buyer interest and earnings momentum (gen6 validated
    this at OOS 0.62+ Calmar).
  - Portfolio vol-targeting (gen9 at 80%+ OOS retention) provides structural
    regime-invariant deleveraging — it doesn't depend on the IS calm-VIX regime.
  - Combining both: near-high filter prevents chasing breakdown names; vol-target
    prevents overexposure in turbulent regimes regardless of VIX level.
  - Distinct from nearhi_momentum_quality (equal-vol inv-vol sizing, no port-vol
    target) and from gen9_sp500_voltarget_skipmon (skip-month, no near-high filter).

Design:
  - Quality filter: price >= 90% of 252d high.
  - Rank by 126d momentum (price return).
  - Hold top-15 above SPY 200d SMA.
  - Inverse-vol weighted (21d realized vol).
  - Portfolio vol-target: scale exposure = clip(12% / 21d_port_vol, 50%, 97%).
  - IEF defensive when SPY below 200d SMA.
  - Biweekly rebalance (every 10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10         # biweekly
MOMENTUM_WINDOW = 126        # ~6 months
HIGH_WINDOW = 252            # 52-week high lookback
NEARHI_THRESHOLD = 0.90      # price must be > 90% of 252d high
VOL_WINDOW = 21              # per-stock inverse-vol + portfolio vol
SPY_TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.12            # 12% annualized portfolio vol target
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 0.97
SQRT252 = float(np.sqrt(252))


class SP500NearHiVolTarget(Strategy):
    """SP500 top-15 by 126d momentum with near-52w-high filter; portfolio
    vol-targeted at 12%; inverse-vol weighted; SPY 200d gate; IEF defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        vol_target: float = VOL_TARGET,
        min_exposure: float = MIN_EXPOSURE,
        max_exposure: float = MAX_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            vol_target=vol_target,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_target = float(vol_target)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)
        # Track portfolio returns for vol-targeting
        self._prev_port_value: float | None = None
        self._port_returns: list[float] = []

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)

        # Track daily portfolio returns for vol-targeting
        if self._prev_port_value is not None and self._prev_port_value > 0:
            daily_ret = (equity - self._prev_port_value) / self._prev_port_value
            self._port_returns.append(daily_ret)
            if len(self._port_returns) > self.vol_window + 5:
                self._port_returns = self._port_returns[-(self.vol_window + 5):]
        self._prev_port_value = equity

        if ctx.idx % self.rebalance_every != 0:
            return []

        if equity <= 0:
            return []

        # Compute portfolio realized vol for vol-targeting
        port_vol_ann = 0.20  # default 20% until we have history
        if len(self._port_returns) >= self.vol_window:
            rv = float(np.std(self._port_returns[-self.vol_window:]))
            if rv > 1e-8 and np.isfinite(rv):
                port_vol_ann = rv * SQRT252

        # Vol-targeted exposure
        if port_vol_ann > 1e-6:
            raw_exposure = self.vol_target / port_vol_ann
        else:
            raw_exposure = self.max_exposure
        exposure = float(np.clip(raw_exposure, self.min_exposure, self.max_exposure))

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            # Defensive: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = exposure
        else:
            # Need lookback for high_window + momentum + vol
            need = self.high_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.high_window - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue

                # 52w-high proximity quality filter
                recent_high = float(col.iloc[-self.high_window:].max())
                if recent_high <= 0 or not np.isfinite(recent_high):
                    continue
                current_price = float(col.iloc[-1])
                nearhi_ratio = current_price / recent_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue

                # 126d momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv_stock = float(np.std(logr))
                if rv_stock <= 1e-6 or not np.isfinite(rv_stock):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv_stock

            if len(scores) < 5:
                # Not enough quality candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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


NAME = "sp500_nearhi_voltarget"
HYPOTHESIS = (
    "SP500 52w-high proximity filter (price > 90% of 252d high) combined with portfolio "
    "vol-targeting (12% annualized vol target via 21d realized portfolio vol, clip 50-97%); "
    "rank by 126d momentum; top-15 above SPY 200d SMA; inverse-vol weighted; IEF defensive; "
    "biweekly rebalance — combines near-high quality screen from gen6 with vol-targeting "
    "mechanism from gen9 for regime-invariant deleveraging"
)

UNIVERSE = _universe

STRATEGY = SP500NearHiVolTarget()
