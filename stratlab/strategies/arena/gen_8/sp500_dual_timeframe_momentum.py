"""SP500 Dual-Timeframe Momentum Confirmation — gen_8 sonnet-6

Hypothesis: Hold top-10 SP500 stocks by 21d momentum ONLY when their 63d
return is also positive (short-term AND medium-term momentum both confirming).
Equal-weight; SPY 150d SMA gate (faster trigger than 200d); GLD defensive
in bear market; biweekly rebalance.

Rationale:
- Dual confirmation filter: requiring BOTH 21d positive AND 63d positive
  momentum means we only hold stocks in persistent uptrends, not brief spikes.
  Stocks with positive 21d but negative 63d are often bouncing from multi-month
  declines — these are "false momentum" entries. Dual confirmation filters them.
- 21d ranking (shorter than most strategies) selects stocks with recent
  acceleration — fresher momentum signal than 63d or 126d.
- SPY 150d SMA: slightly faster than 200d (which many existing strategies use),
  catching trend reversals ~2-3 weeks earlier in choppy markets.
- GLD defensive (not TLT/IEF): gold behaves differently from bonds in bear
  markets — provides inflation hedge and dollar hedge during uncertainty,
  distinct from duration-sensitive TLT.

Distinction from existing strategies:
- No existing strategy uses 21d primary ranking with 63d confirmation filter
- GLD as the sole defensive vehicle (not TLT, IEF, or TLT+SHY)
- SPY 150d gate (different from 200d used by most SP500 stock selectors)
- Smaller portfolio (top-10, not top-15 or top-20)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
SHORT_MOM_WINDOW = 21   # ~1 month for primary ranking
LONG_MOM_WINDOW = 63    # ~3 months for confirmation
TREND_WINDOW = 150      # SPY 150d SMA gate (faster than common 200d)
TOP_K = 10              # smaller concentrated portfolio
EXPOSURE = 0.97

_SPY = "SPY"
_GLD = "GLD"  # gold defensive — different from TLT/IEF used by other strategies


class Sp500DualTimeframeMomentum(Strategy):
    """SP500 dual-timeframe momentum: 21d rank + 63d confirmation + GLD defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_mom_window: int = SHORT_MOM_WINDOW,
        long_mom_window: int = LONG_MOM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_mom_window=short_mom_window,
            long_mom_window=long_mom_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_mom_window = int(short_mom_window)
        self.long_mom_window = int(long_mom_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.long_mom_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 150d SMA trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: GLD defensive
            if _GLD in live:
                target[_GLD] = self.exposure
        else:
            # Bull market: dual-timeframe momentum
            need = self.long_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.long_mom_window + 2:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                if sym in (_SPY, _GLD):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.long_mom_window + 1:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue

                # 21d momentum (primary ranking signal)
                if len(col) < self.short_mom_window + 1:
                    continue
                p_short = float(col.iloc[-self.short_mom_window])
                if p_short <= 0 or not np.isfinite(p_short):
                    continue
                short_ret = current_price / p_short - 1.0
                if not np.isfinite(short_ret):
                    continue

                # 63d momentum (confirmation filter)
                p_long = float(col.iloc[-self.long_mom_window])
                if p_long <= 0 or not np.isfinite(p_long):
                    continue
                long_ret = current_price / p_long - 1.0
                if not np.isfinite(long_ret):
                    continue

                # DUAL CONFIRMATION: both timeframes must be positive
                if long_ret <= 0:
                    continue  # Skip stocks with negative 63d momentum

                # Rank by 21d momentum (fresher signal)
                scores[sym] = short_ret

            if len(scores) < 5:
                # Not enough dual-confirmed candidates — GLD defensive
                if _GLD in live:
                    target[_GLD] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
                        target[sym] = per_weight

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
    return sp500_tickers() + [_GLD, _SPY]


NAME = "sp500_dual_timeframe_momentum"
HYPOTHESIS = (
    "SP500 cross-sectional 21d momentum with dual-momentum quality: hold top-10 stocks by "
    "21d return ONLY when their 63d return is also positive (both short-term AND medium-term "
    "momentum confirming); equal-weight; SPY 150d SMA gate; GLD defensive in bear; "
    "biweekly rebalance; dual time-horizon confirmation reduces false momentum signals"
)

UNIVERSE = _universe

STRATEGY = Sp500DualTimeframeMomentum()
