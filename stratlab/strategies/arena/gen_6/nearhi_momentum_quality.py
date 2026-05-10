"""Near-52w-high momentum quality filter on SP500.

Hypothesis: hold top-15 SP500 stocks with highest 126d momentum AND
price-to-52w-high ratio > 0.80 (near-high quality filter), inverse-vol
weighted; SPY 200d SMA gate; TLT defensive; monthly rebalance.

Rationale: The price-to-52w-high ratio acts as a quality filter — stocks
near their 52w high show persistent buyer interest and earnings momentum.
Combining it with raw momentum filters out "lottery" momentum names that
spike once then collapse. Inverse-vol weighting reduces concentration risk
in high-beta winners. The 200d SMA gate avoids holding quality growth stocks
in bear markets.

Distinction from existing strategies:
  - Uses near-52w-high proximity as a quality filter (not in any existing strategy)
  - Inverse-vol sizing like xsect_12m_invvol but different gate (200d SMA only)
    and 126d momentum (not 6-1 month skip)
  - Monthly rebalance, TLT defensive
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
MOMENTUM_WINDOW = 126      # ~6 months
HIGH_WINDOW = 252          # 52-week high lookback
NEARHI_THRESHOLD = 0.80    # price must be > 80% of 52w high
VOL_WINDOW = 20            # for inverse-vol weights
TOP_K = 15
TREND_WINDOW = 200
EXPOSURE = 0.97


class NearHiMomentumQuality(Strategy):
    """Near-52w-high momentum quality: top-15 SP500 stocks near 52w high with
    strong 126d momentum; inverse-vol sized; SPY 200d gate; TLT defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA for regime gate
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
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Need enough history for 52w high + momentum + vol
            need = self.high_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue

                # Price-to-52w-high ratio (quality filter)
                recent_252 = col.iloc[-self.high_window:]
                w52_high = float(recent_252.max())
                if w52_high <= 0 or not np.isfinite(w52_high):
                    continue
                current_price = float(col.iloc[-1])
                nearhi_ratio = current_price / w52_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue  # Skip stocks far from their 52w high

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

                # Inverse-vol weighting
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
                # Not enough candidates — fall back to TLT
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


NAME = "nearhi_momentum_quality"
HYPOTHESIS = (
    "Earnings-momentum quality: hold top-15 SP500 stocks with highest 126d momentum AND "
    "price-to-52w-high ratio > 0.80 (near-high quality filter), inverse-vol weighted; "
    "SPY 200d SMA gate; TLT defensive; monthly rebalance"
)

UNIVERSE = _universe

STRATEGY = NearHiMomentumQuality()
