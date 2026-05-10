"""SP500 near-high momentum with inverse-vol weighting strategy.

Hypothesis: SP500 stocks that are near their 52-week high (price >85% of
252d high) AND have positive 63d momentum represent high-quality momentum
names with institutional sponsorship. Weighting inversely by realized
volatility gives more weight to steadier risers over momentum spikes.

Signal:
  - Quality filter: current price > 85% of 252-day rolling high
  - Momentum filter: 63d return > 0
  - Ranking: by composite (0.6 * 63d_return + 0.4 * proximity_to_high)
  - Weighting: inversely by 20d realized volatility (lower vol = bigger weight)
  - Hold top-15 qualifying stocks
  - Gate: SPY above 100d SMA (shorter gate = less defensive drag)
  - Defensive: IEF (intermediate treasury, less duration risk than TLT)
  - Biweekly rebalance (every 10 bars)

Rationale: The 85% proximity filter is less restrictive than 95% but still
ensures only uptrending stocks qualify. Inverse-vol weighting concentrates
capital in lower-volatility momentum stocks (e.g., steady compounders
rather than volatile momentum spikes). Using SPY 100d SMA (not 200d) means
less time in defensive IEF — faster re-entry after corrections.

Diversification vs leaderboard:
  - gen6_sp500_52wk_high_breakout: uses 95% proximity + 200d SMA + equal weight
  - gen5_xsect_12m_invvol_goldencross: 200d gate + 126d skip-21d momentum
  - This strategy: 100d gate + 85% proximity + 63d momentum + inv-vol weighting
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 63    # 3-month momentum
HIGH_WINDOW = 252       # 52-week high lookback
PROXIMITY_MIN = 0.85    # must be within 85% of 252d high (15% below = excluded)
VOL_WINDOW = 20         # inverse-vol weighting
TOP_K = 15
TREND_WINDOW = 100      # SPY 100d SMA gate (shorter = less defensive)
EXPOSURE = 0.97


class SP500NearHighInvVol(Strategy):
    """SP500 near-52wk-high + positive 63d mom, inverse-vol weighted, SPY 100d SMA gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        proximity_min: float = PROXIMITY_MIN,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            proximity_min=proximity_min,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.proximity_min = float(proximity_min)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + self.vol_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 100d SMA market regime gate
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
            # Bear market: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Bull market: near-high + momentum + inverse-vol
            need = self.high_window + self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.high_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                min_needed = max(self.high_window, self.momentum_window, self.vol_window + 1)
                if len(col) < min_needed:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue

                # 52-week high proximity filter
                rolling_high = float(col.iloc[-self.high_window:].max())
                if rolling_high <= 0:
                    continue
                proximity_ratio = current_price / rolling_high  # 1.0 = at high
                if proximity_ratio < self.proximity_min:
                    continue  # more than 15% below 52wk high → skip

                # Momentum filter: 63d return must be positive
                mom = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom) or mom <= 0:
                    continue

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                log_rets = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(log_rets))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                # Composite score for ranking
                composite = 0.6 * mom + 0.4 * (proximity_ratio - self.proximity_min)
                scores[sym] = composite
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Fallback to IEF if too few qualifying stocks
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
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_nearhigh_invvol"
HYPOTHESIS = (
    "SP500 near-high momentum with inverse-vol weighting: hold top-15 SP500 stocks "
    "with price above 85% of 252d high (quality filter) AND positive 63d momentum, "
    "weighted inversely by 20d realized vol; SPY 100d SMA gate; IEF defensive; "
    "biweekly rebalance; distinct from 200d-SMA equal-weight approaches"
)

UNIVERSE = _universe

STRATEGY = SP500NearHighInvVol()
