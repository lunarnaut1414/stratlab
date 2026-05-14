"""SP500 Quality-Tilt Momentum — gen_8 sonnet-4

Hypothesis: Rank SP500 stocks by 126d momentum, apply a dual quality gate
(price >= 85% of 252d high AND above 100d SMA). Hold top-20 inverse-vol
weighted. SPY 200d SMA bear gate to IEF. Biweekly rebalance.

Rationale: Pure momentum captures recent winners; adding the near-52w-high
filter restricts picks to stocks in sustained structural uptrends (no dead-cat
bounces from oversold conditions). The 100d SMA individual-stock trend gate
further ensures the holding is in its own intermediate-term uptrend.
Inverse-vol weighting gives smaller positions to volatile, high-momentum
names — smoother performance than equal-weight.

Distinction from existing strategies:
- gen6_nearhi_momentum_quality uses 80% near-high threshold, equal-weight,
  monthly rebalance. This uses 85% threshold, inverse-vol weight, biweekly.
- gen7_sp500_126d_stock_50sma_goldencross uses 50d individual stock SMA
  (shorter) and equal-weight. This uses 100d SMA (intermediate trend) and
  inverse-vol weighting.
- Pure momentum strategies don't have the quality/structural-uptrend angle.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bi-weekly
MOMENTUM_WINDOW = 126     # ~6 months
HIGH_WINDOW = 252         # 52-week high lookback
NEARHI_THRESHOLD = 0.85   # price must be >= 85% of 252d high
STOCK_SMA = 100           # individual stock intermediate trend
TREND_WINDOW = 200        # SPY market-wide trend gate
TOP_K = 20
EXPOSURE = 0.97
VOL_WINDOW = 21           # vol for inverse-vol sizing


class QualityTiltMomentum(Strategy):
    """SP500 quality-filtered momentum with inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        stock_sma: int = STOCK_SMA,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        vol_window: int = VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            stock_sma=stock_sma,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            vol_window=vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.stock_sma = int(stock_sma)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.vol_window = int(vol_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.high_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma200 = float(spy_close.iloc[-self.trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        bull = spy_price > spy_sma200

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
            # Get enough window for all signals
            need = max(self.high_window, self.momentum_window, self.stock_sma, self.vol_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            vols: dict[str, float] = {}

            for sym in prices.columns:
                # Skip non-tradeable symbols
                if sym.startswith("^") or sym.endswith("=F") or sym.endswith("=X"):
                    continue

                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0:
                    continue

                # Quality gate 1: above individual 100d SMA
                if len(col) >= self.stock_sma:
                    stock_sma_val = float(col.iloc[-self.stock_sma:].mean())
                    if current_price <= stock_sma_val:
                        continue

                # Quality gate 2: near 252d high (>= 85% of high)
                hi_window_len = min(self.high_window, len(col))
                rolling_high = float(col.iloc[-hi_window_len:].max())
                if rolling_high <= 0:
                    continue
                nearhi_ratio = current_price / rolling_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue

                # 126d momentum
                if len(col) < self.momentum_window:
                    continue
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                ret = current_price / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                scores[sym] = ret

                # Compute volatility for sizing
                if len(col) >= self.vol_window + 1:
                    daily_rets = col.iloc[-self.vol_window:].pct_change().dropna()
                    if len(daily_rets) >= 5:
                        vol = float(daily_rets.std())
                        vols[sym] = vol if vol > 0 else 0.01
                    else:
                        vols[sym] = 0.02
                else:
                    vols[sym] = 0.02

            if len(scores) < 5:
                # Fallback to defensive if not enough quality candidates
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Inverse-vol weights
                inv_vols = {sym: 1.0 / vols.get(sym, 0.02) for sym in ranked}
                total_inv_vol = sum(inv_vols.values())
                if total_inv_vol <= 0:
                    # Fallback to equal-weight
                    weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = weight
                else:
                    for sym in ranked:
                        target[sym] = (inv_vols[sym] / total_inv_vol) * self.exposure

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


NAME = "quality_tilt_momentum"
HYPOTHESIS = (
    "SP500 quality-tilt momentum: rank SP500 stocks by 126d momentum, apply quality gate "
    "(price at least 85% of 252d high AND above 100d SMA), hold top-20 inverse-vol weighted; "
    "SPY 200d SMA bear gate to IEF; biweekly rebalance; captures stocks with sustained "
    "upward drift not just recent winners"
)

UNIVERSE = _universe

STRATEGY = QualityTiltMomentum()
