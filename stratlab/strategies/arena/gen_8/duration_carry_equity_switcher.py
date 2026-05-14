"""Duration-Carry Equity Switcher — gen_8 sonnet-7

Hypothesis: Use TLT vs IEF 21d return spread as a rate-regime signal to
dynamically route equity exposure between:

  - TLT outperforms IEF (falling rates, duration-positive): hold QQQ 97%
    Growth stocks benefit most from lower long-term rates — QQQ captures
    this tech/growth duration premium.
  - IEF outperforms TLT (neutral/rising rates): hold top-20 SP500 stocks
    by 63d momentum (broad equity momentum, not rate-sensitive growth)
  - SPY below 200d SMA (outer bear gate): hold IEF (intermediate duration,
    balanced defensive)

Rationale:
  The TLT/IEF relative performance is a market-based real-time measure of
  duration demand and rate direction — more responsive than TNX-level gates,
  more equity-relevant than yield-curve slope, and different from VIX or
  credit signals.

  When rates are falling (TLT outperforming IEF), growth stocks (captured
  by QQQ) benefit the most from lower discount rates on future earnings.
  When rates are flat or rising, no particular duration advantage accrues to
  growth — a broader SP500 momentum portfolio captures equity risk premium
  more neutrally.

  This is distinct from:
  - gen7_yield_curve_slope_rotation (uses TNX-IRX level spread, not ETF returns)
  - gen7_opus1_longend_curve_mtum_rotation (uses TYX-TNX long-end signal)
  - gen6_bond_termstruct_curve_rotation (pure bond allocator, no equity)
  - All SP500 stock-selection strategies (uses QQQ as the rate-sensitive proxy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly — more responsive to rate signal
MOMENTUM_WINDOW = 63       # for SP500 stock ranking
RATE_WINDOW = 21           # TLT vs IEF return spread window
TREND_WINDOW = 200         # SPY market gate
TOP_K = 20                 # in neutral rate regime
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_IEF = "IEF"


class DurationCarryEquitySwitcher(Strategy):
    """Rate-regime equity router: QQQ in falling-rate, SP500 momentum in neutral."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rate_window: int = RATE_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rate_window=rate_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rate_window = int(rate_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY outer trend gate
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
        bull_market = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull_market:
            # Bear market: defensive IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Compute TLT vs IEF 21d return spread
            tlt_outperforms = False
            try:
                tlt_hist = ctx.history(_TLT)
                ief_hist = ctx.history(_IEF)
                if (tlt_hist is not None and ief_hist is not None and
                        len(tlt_hist) >= self.rate_window + 2 and
                        len(ief_hist) >= self.rate_window + 2):
                    tlt_close = tlt_hist["close"].dropna()
                    ief_close = ief_hist["close"].dropna()
                    if (len(tlt_close) >= self.rate_window + 1 and
                            len(ief_close) >= self.rate_window + 1):
                        tlt_ret = float(tlt_close.iloc[-1] / tlt_close.iloc[-self.rate_window] - 1.0)
                        ief_ret = float(ief_close.iloc[-1] / ief_close.iloc[-self.rate_window] - 1.0)
                        if np.isfinite(tlt_ret) and np.isfinite(ief_ret):
                            tlt_outperforms = tlt_ret > ief_ret
            except Exception:
                pass

            if tlt_outperforms:
                # Falling-rate regime: hold QQQ (growth/duration beneficiary)
                if _QQQ in live:
                    target[_QQQ] = self.exposure
                elif _SPY in live:
                    # QQQ not in live closes (unusual), fall back to SPY
                    target[_SPY] = self.exposure
            else:
                # Neutral/rising-rate regime: hold top-K SP500 momentum stocks
                need = self.momentum_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 1:
                    if _IEF in live:
                        target[_IEF] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _QQQ, _TLT, _IEF):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + 1:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret

                    if len(scores) < 5:
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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _QQQ]


NAME = "duration_carry_equity_switcher"
HYPOTHESIS = (
    "Duration-carry equity switcher: use TLT vs IEF 21d return spread as rate-regime signal; "
    "when TLT outperforms IEF (falling rates, duration-positive environment), hold QQQ 97% "
    "(growth stocks benefit from rate tailwind); when IEF outperforms TLT (neutral/rising "
    "rates), hold top-20 SP500 stocks by 63d momentum equal-weight; SPY 200d SMA outer bear "
    "gate sends to IEF; weekly rebalance — rate-duration regime determines equity allocation "
    "vehicle, not just on/off switch"
)

UNIVERSE = _universe

STRATEGY = DurationCarryEquitySwitcher()
