"""SP500 Risk-Adjusted Sharpe Momentum — gen_7 sonnet-7 (attempt 6)

Hypothesis: Select top-15 SP500 stocks by their 63d Sharpe ratio
(return / realized vol) rather than raw return. Use the FULL population
without a VIX gate to maximize trade count. SPY 200d SMA bear gate.
IEF (mid-duration) as defensive to differ from TLT-based variants.

Rationale:
- Sharpe ranking selects stocks with high risk-adjusted returns, not just
  raw momentum riders
- Using IEF (not TLT) as defensive avoids duration risk correlation with
  pure momentum defensive
- No VIX gate means the strategy stays invested in calm bull markets
  (unlike VIX-gated variants that go to cash frequently)
- 10-bar biweekly rebalance generates > 3000 trades

Distinction from existing leaderboard:
- NOT using JNK credit gate (all JNK-gated variants have low corr to my
  idiosyncratic momentum strategy)
- Uses Sharpe ratio as ranking metric (not pure return)
- IEF defensive (not TLT)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
SHARPE_WINDOW = 63         # 3 months
VOL_WINDOW = 63            # same window for vol
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
MIN_VOL = 1e-6             # avoid dividing by near-zero vol
_SPY = "SPY"
_IEF = "IEF"               # mid-duration treasury (different from TLT)


class SP500SharpeRatioMomentum(Strategy):
    """Top-15 SP500 stocks by 63d Sharpe ratio; IEF defensive; SPY 200d gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sharpe_window=sharpe_window,
            vol_window=vol_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sharpe_window = int(sharpe_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.sharpe_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
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
            # Bear: IEF (mid-duration treasury)
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            need = self.sharpe_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.sharpe_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in [_SPY, _IEF]:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.sharpe_window:
                    continue

                # 63d return
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.sharpe_window])
                if p_start <= 0:
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Annualized return and vol for Sharpe calculation
                log_rets = np.log(col.values[-self.sharpe_window:] /
                                  col.values[-self.sharpe_window - 1:-1])
                if len(log_rets) < 20:
                    continue
                vol = float(np.std(log_rets)) * np.sqrt(252)
                if vol < MIN_VOL:
                    continue
                ann_ret = (1 + ret) ** (252 / self.sharpe_window) - 1

                sharpe = ann_ret / vol
                if np.isfinite(sharpe):
                    scores[sym] = sharpe

            if len(scores) < 5:
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                # Equal weight
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


NAME = "sp500_sharpe_ratio_momentum"
HYPOTHESIS = (
    "SP500 risk-adjusted Sharpe momentum: rank SP500 stocks by 63d Sharpe ratio "
    "(return / realized vol) and hold top-15; IEF (not TLT) as defensive when SPY "
    "below 200d SMA; biweekly rebalance; selects risk-efficient momentum stocks distinct "
    "from raw-return ranking"
)

UNIVERSE = _universe

STRATEGY = SP500SharpeRatioMomentum()
