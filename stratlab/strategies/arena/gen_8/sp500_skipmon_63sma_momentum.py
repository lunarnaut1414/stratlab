"""SP500 Skip-Month Momentum with 63d SMA Stock-Level Filter — gen_8 sonnet-7

Hypothesis: Rank SP500 stocks by 126d-skip-21d return (Jegadeesh-Titman
skip-month momentum: use 126 days lookback excluding the most recent 21 days).
Hold only stocks that are also above their own 63d SMA (intermediate trend
confirmation). Inverse-vol weighted. SPY 200d SMA market gate; IEF defensive.

Rationale:
  - The skip-month (126d-skip-21d) avoids short-term mean-reversion — stocks
    with strong momentum often give back some returns over the most recent
    1-month window before resuming, so skipping the most recent month selects
    stocks in sustained 5-month+ uptrends rather than 1-month spikes.
  - A per-stock 63d SMA gate (medium-term, not 200d long-term) catches stocks
    in confirmed intermediate-term uptrends. Unlike 200d SMA or 50d SMA filters
    used by existing strategies, the 63d SMA is an intermediate filter that
    balances signal timeliness with noise rejection.
  - Inverse-vol sizing ensures large-momentum-but-volatile stocks don't
    dominate the portfolio.

Differentiators vs leaderboard:
  - Skip-month momentum (126d-21d) vs simple 63d or 126d (full window)
  - Per-stock 63d SMA gate vs 50d or 200d SMA gates used by others
  - IEF (not TLT) as defensive — IEF is shorter duration and less rate-sensitive
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_LONG = 126             # long lookback (6 months)
MOM_SKIP = 21              # skip most recent month to avoid reversal
STOCK_SMA = 63             # per-stock intermediate trend filter
TREND_WINDOW = 200         # market-wide SPY gate
VOL_WINDOW = 21            # for inverse-vol sizing
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"


class SP500SkipMonMomentum(Strategy):
    """Skip-month momentum (126d-21d) with 63d per-stock SMA trend gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        stock_sma: int = STOCK_SMA,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            stock_sma=stock_sma,
            trend_window=trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.stock_sma = int(stock_sma)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY market trend gate
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
            # Defensive: IEF (shorter duration than TLT)
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Compute skip-month momentum scores
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long + 2:
                return []

            scores: dict[str, float] = {}
            vols: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _IEF):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.mom_long + 1:
                    continue

                # Skip-month momentum: return from T-126 to T-21
                # (exclude most recent 21 trading days)
                price_at_long = float(col.iloc[-self.mom_long])
                price_at_skip = float(col.iloc[-self.mom_skip])
                if price_at_long <= 0 or price_at_skip <= 0:
                    continue
                skipmon_ret = float(price_at_skip / price_at_long - 1.0)
                if not np.isfinite(skipmon_ret):
                    continue

                # Per-stock 63d SMA gate: stock must be above intermediate trend
                if len(col) < self.stock_sma + 1:
                    continue
                stock_sma_val = float(col.iloc[-self.stock_sma:].mean())
                current_price = float(col.iloc[-1])
                if current_price <= stock_sma_val:
                    continue  # below 63d SMA — skip this stock

                scores[sym] = skipmon_ret

                # Inverse-vol sizing
                daily_rets = col.pct_change().dropna()
                if len(daily_rets) >= self.vol_window:
                    rv = float(daily_rets.iloc[-self.vol_window:].std())
                    vols[sym] = max(rv, 1e-6)

            if len(scores) < 5:
                # Not enough qualified candidates
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Inverse-vol weighting
                inv_vols = {}
                for sym in ranked:
                    vol = vols.get(sym, 0.02)
                    inv_vols[sym] = 1.0 / max(vol, 1e-6)
                total_inv = sum(inv_vols.values())
                if total_inv <= 0:
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight
                else:
                    for sym in ranked:
                        if sym in live:
                            target[sym] = self.exposure * (inv_vols[sym] / total_inv)

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
    return sp500_tickers() + [_IEF, _SPY]


NAME = "sp500_skipmon_63sma_momentum"
HYPOTHESIS = (
    "SP500 skip-month momentum with per-stock 63d SMA trend confirmation: rank SP500 stocks "
    "by 126d-skip-21d return (classic Jegadeesh-Titman skip-month to avoid short-term "
    "reversal); hold top-15 stocks that are also above their own 63d SMA (intermediate "
    "trend confirmed); inverse-vol weighted; SPY 200d SMA market gate; IEF defensive; "
    "biweekly rebalance — skip-month + stock 63d SMA combination not present in leaderboard"
)

UNIVERSE = _universe

STRATEGY = SP500SkipMonMomentum()
