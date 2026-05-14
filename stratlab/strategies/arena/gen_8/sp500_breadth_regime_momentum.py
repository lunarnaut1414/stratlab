"""SP500 Momentum with Market-Breadth Regime Sizing — gen_8 sonnet-6

Hypothesis: Use the RSP/SPY 40d return spread as a market-breadth regime signal
to dynamically adjust between aggressive (broad participation) and conservative
(narrow leadership) momentum. Broad participation (RSP>SPY on 40d returns):
hold top-20 SP500 stocks by 42d momentum at full 97% exposure. Narrow
leadership (RSP<SPY): hold top-10 stocks at 65% exposure. SPY 200d SMA gate;
IEF defensive in bear; biweekly rebalance.

Rationale:
- RSP (equal-weight S&P500) vs SPY (cap-weight) spread measures market breadth:
  when equal-weight outperforms cap-weight, broad participation supports a
  higher-conviction momentum portfolio.
- When only mega-caps lead (SPY>>RSP), momentum is narrowly concentrated and
  likely to mean-revert — reduce exposure and portfolio size.
- This is a RISK-SIZING strategy using breadth as a sizing signal, not a
  rotation signal. All exposure stays in SP500 stocks (not ETF rotation).
- Different from gen_7's breadth-failed attempts: those rotated TO sector ETFs
  based on breadth. This stays in SP500 stocks but resizes the portfolio.

Distinction from existing strategies:
- RSP/SPY breadth drives POSITION SIZING and PORTFOLIO SIZE (not ETF allocation)
- Two-tier momentum (top-20 in broad market, top-10 in narrow market) is novel
- Broader portfolio in broad-market regimes reduces concentration risk naturally
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 42      # 42d momentum ranking
BREADTH_WINDOW = 40       # RSP/SPY 40d return for breadth signal
TREND_WINDOW = 200        # SPY 200d SMA gate
TOP_K_BROAD = 20          # top-20 stocks in broad market
TOP_K_NARROW = 10         # top-10 stocks in narrow market
EXPOSURE_BROAD = 0.97     # full exposure in broad breadth
EXPOSURE_NARROW = 0.65    # reduced exposure in narrow breadth

_SPY = "SPY"
_RSP = "RSP"   # Invesco S&P 500 Equal Weight ETF
_IEF = "IEF"  # IEF defensive in bear


class Sp500BreadthRegimeMomentum(Strategy):
    """SP500 momentum with RSP/SPY breadth signal for portfolio size and exposure."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k_broad: int = TOP_K_BROAD,
        top_k_narrow: int = TOP_K_NARROW,
        exposure_broad: float = EXPOSURE_BROAD,
        exposure_narrow: float = EXPOSURE_NARROW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            breadth_window=breadth_window,
            trend_window=trend_window,
            top_k_broad=top_k_broad,
            top_k_narrow=top_k_narrow,
            exposure_broad=exposure_broad,
            exposure_narrow=exposure_narrow,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.breadth_window = int(breadth_window)
        self.trend_window = int(trend_window)
        self.top_k_broad = int(top_k_broad)
        self.top_k_narrow = int(top_k_narrow)
        self.exposure_broad = float(exposure_broad)
        self.exposure_narrow = float(exposure_narrow)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.breadth_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
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
            # Bear market: IEF defensive
            if _IEF in live:
                target[_IEF] = self.exposure_broad
        else:
            # Measure market breadth: RSP vs SPY 40d return
            broad_market = True  # default to broad if RSP unavailable
            try:
                rsp_hist = ctx.history(_RSP)
                spy_hist2 = ctx.history(_SPY)
                if (len(rsp_hist) >= self.breadth_window + 2 and
                        len(spy_hist2) >= self.breadth_window + 2):
                    rsp_close = rsp_hist["close"].dropna()
                    spy_close2 = spy_hist2["close"].dropna()
                    if (len(rsp_close) >= self.breadth_window + 1 and
                            len(spy_close2) >= self.breadth_window + 1):
                        rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                        spy_ret = float(spy_close2.iloc[-1] / spy_close2.iloc[-self.breadth_window] - 1.0)
                        # Broad market = equal-weight outperforms cap-weight
                        broad_market = rsp_ret > spy_ret
            except KeyError:
                pass  # RSP not available, default to broad

            # Set portfolio parameters based on breadth regime
            top_k = self.top_k_broad if broad_market else self.top_k_narrow
            exposure = self.exposure_broad if broad_market else self.exposure_narrow

            # Get momentum scores
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _RSP, _IEF):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 1:
                    continue
                current_price = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(current_price):
                    continue
                ret = current_price / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 5:
                if _IEF in live:
                    target[_IEF] = self.exposure_broad
            else:
                k = min(top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = exposure / len(ranked)
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
    return sp500_tickers() + [_IEF, _SPY, _RSP]


NAME = "sp500_breadth_regime_momentum"
HYPOTHESIS = (
    "SP500 42d momentum with RSP/SPY 40d return spread as market-breadth regime signal: "
    "broad participation (RSP>SPY) hold top-20 stocks at 97% exposure; "
    "narrow leadership (RSP<SPY) hold top-10 stocks at 65% exposure; "
    "SPY 200d SMA gate; IEF defensive in bear; biweekly rebalance; "
    "breadth signal adjusts portfolio SIZE and EXPOSURE rather than rotating to ETFs"
)

UNIVERSE = _universe

STRATEGY = Sp500BreadthRegimeMomentum()
