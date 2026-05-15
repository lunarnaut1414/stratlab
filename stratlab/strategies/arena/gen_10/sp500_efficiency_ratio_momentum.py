"""SP500 momentum with Price Efficiency Ratio quality screen.

Hypothesis (sonnet-8, gen_10):
    Rank SP500 stocks by 126d momentum, but only include stocks with a
    Price Efficiency Ratio (ER) >= 0.5 computed over 20 bars. The ER =
    |net directional move| / sum(|daily moves|) measures how directionally
    efficient recent price movement is. ER near 1.0 = clean trending; ER
    near 0 = whipsaw/noise. This filters out "noisy momentum" stocks that
    happen to have a large 6-month return but are actually oscillating wildly.
    Inverse-vol weighted. SPY 200d SMA gate to IEF. Biweekly rebalance.

Rationale:
    - The Efficiency Ratio was popularized by Perry Kaufman for adaptive moving
      averages. Applied as a stock filter, it selects stocks whose price has
      moved directionally (clean uptrend) rather than noisily (lottery-style).
    - High-ER stocks in an uptrend are more likely to STAY trending (their
      upward movement reflects persistent buying, not noise).
    - Low-ER momentum stocks are more fragile — they happen to have a positive
      6-month return via a volatile path that could reverse.
    - Combined with 126d momentum ranking, ER acts as a persistence/quality filter
      orthogonal to RSI (oversold avoidance), near-52w-high (level-based quality),
      and stochastic (short-cycle position).

Efficiency Ratio definition:
    ER = |close_t - close_{t-N}| / sum_{i=1}^{N} |close_i - close_{i-1}|
    Range: [0, 1]. 1 = perfectly directional. 0 = perfectly oscillating.
    Threshold: ER >= 0.50 means directional move is at least half of total movement.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 126      # 6-month momentum
ER_WINDOW = 20             # 20-bar efficiency ratio
ER_FLOOR = 0.50            # minimum efficiency ratio to be eligible
VOL_WINDOW = 21            # inverse-vol weighting lookback
SPY_TREND_WINDOW = 200     # 200d SMA outer gate
TOP_K = 15
EXPOSURE = 0.97


def _efficiency_ratio(prices: np.ndarray, window: int) -> float:
    """Compute Price Efficiency Ratio over last `window` bars.

    ER = |end_price - start_price| / sum(|daily_change|)
    Returns NaN if insufficient data or zero total movement.
    """
    if len(prices) < window + 1:
        return float("nan")
    recent = prices[-(window + 1):]
    net_move = abs(float(recent[-1]) - float(recent[0]))
    daily_moves = np.abs(np.diff(recent.astype(float)))
    total_path = float(np.sum(daily_moves))
    if total_path < 1e-9:
        return 1.0  # flat = perfectly "efficient" (no noise)
    er = net_move / total_path
    return float(er)


class SP500EfficiencyRatioMomentum(Strategy):
    """SP500 126d momentum with Efficiency Ratio >= 0.5 quality filter;
    inverse-vol sized; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        er_window: int = ER_WINDOW,
        er_floor: float = ER_FLOOR,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            er_window=er_window,
            er_floor=er_floor,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.er_window = int(er_window)
        self.er_floor = float(er_floor)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.er_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = self.momentum_window + self.er_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.er_window:
                    continue

                # Efficiency ratio quality filter
                er = _efficiency_ratio(col.values, self.er_window)
                if not np.isfinite(er) or er < self.er_floor:
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
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe

NAME = "sp500_efficiency_ratio_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with price efficiency ratio quality screen: only include "
    "stocks where 20d Efficiency Ratio >= 0.50 (directional move / total path length — filters "
    "noisy/whipsaw momentum); inverse-vol weighted; SPY 200d SMA gate to IEF defensive; "
    "biweekly rebalance — efficiency ratio filters out chaotic high-momentum names that revert, "
    "selecting genuinely trending stocks"
)

STRATEGY = SP500EfficiencyRatioMomentum()
