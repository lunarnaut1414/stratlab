"""SP500 Risk-Adjusted Momentum — gen_8 sonnet-10

Hypothesis: Rank SP500 stocks by their 63d return divided by 63d realized
volatility (annualized Sharpe-like score). Hold top-15 stocks above their
200d SMA. This differs from raw-return momentum by penalizing high-vol
names that had strong raw returns — it selects persistent, smooth risers.

Rationale: Raw momentum over-weights high-beta names in bull markets.
A return-per-unit-vol score identifies stocks with genuinely consistent
upward drift, not just lucky high-beta rides. This should be lower-corr
to the leaderboard's raw-momentum leaders (idiosyncratic, 126d, etc.).

SPY 200d SMA gate: avoid individual stocks in bear regimes.
IEF as defensive (less correlated to TLT-heavy defensive positions on board).
Biweekly rebalance: 10 bars.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # ~3 months
TREND_WINDOW = 200        # 200d SMA gate
VOL_WINDOW = 63           # same window for vol as momentum
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"
_TRADING_DAYS_PER_YEAR = 252


class SP500RiskAdjustedMomentum(Strategy):
    """SP500 momentum ranked by return/volatility (Sharpe-like score)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
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

        # SPY trend gate
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
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Need window for both vol and momentum computation
            need = self.trend_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym == _SPY or sym == _IEF:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # 63d raw return
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(ret) or ret <= 0:
                    # Only score positive-momentum stocks
                    continue

                # 63d realized vol (annualized) from daily log-returns
                recent = col.iloc[-self.vol_window:]
                log_rets = np.log(recent.values[1:] / recent.values[:-1])
                if len(log_rets) < 20:
                    continue
                ann_vol = float(np.std(log_rets) * np.sqrt(_TRADING_DAYS_PER_YEAR))
                if ann_vol < 0.01:
                    continue

                # Per-stock 200d SMA filter
                if len(col) >= self.trend_window:
                    stock_sma = float(col.iloc[-self.trend_window:].mean())
                    stock_price = float(col.iloc[-1])
                    if stock_price < stock_sma:
                        continue

                # Sharpe-like score: annualized return / annualized vol
                ann_ret = (1 + ret) ** (_TRADING_DAYS_PER_YEAR / self.momentum_window) - 1
                sharpe_like = ann_ret / ann_vol
                if np.isfinite(sharpe_like):
                    scores[sym] = sharpe_like

            if len(scores) < 5:
                # Not enough candidates — hold IEF
                if _IEF in live:
                    target[_IEF] = self.exposure
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
    return sp500_tickers() + [_IEF, _SPY]


NAME = "sp500_riskadjusted_momentum"
HYPOTHESIS = (
    "SP500 risk-adjusted momentum: rank SP500 stocks by 63d return divided by 63d realized "
    "volatility (Sharpe-like score), hold top-15 stocks above their 200d SMA; equal-weight; "
    "SPY 200d SMA gate; IEF defensive in bear; biweekly rebalance; targets stocks with high "
    "return-per-unit-risk not just raw momentum"
)

UNIVERSE = _universe

STRATEGY = SP500RiskAdjustedMomentum()
