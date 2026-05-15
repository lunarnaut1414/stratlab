"""gen_9 sonnet-1 — IAU/TLT Inflation Regime Signal → SP500 Momentum

Hypothesis: Use the relative performance of gold (IAU) vs long bonds (TLT)
on a 42d basis as an inflation/deflation regime signal.
- IAU 42d return > TLT 42d return (gold outperforming bonds) → inflation regime →
  hold top-15 SP500 stocks by 63d momentum above 200d SMA (equities outperform
  in inflation); equal-weight; SPY 200d outer bear gate overrides to TLT.
- TLT outperforming IAU → deflationary/safety regime →
  hold TLT 60% + IAU 37% (bond safety + gold hedge).

Rationale:
- Gold vs bonds spread is a proxy for inflation expectations / real rates.
  When gold is rising faster than long bonds, real rates are falling (inflation
  rising or nominal yields declining) — historically favorable for equities.
  When bonds are outperforming gold, real rates are rising or flight-to-safety is
  on — reduce equity exposure.
- This inflation/deflation signal creates completely different regime timing
  compared to VIX, JNK, TNX, VWO/VEA, or UUP-based gates on the leaderboard.
- SP500 momentum stock selection in the risk-on branch provides high trade count.

Coverage (all cover IS 2010-2018):
  IAU (2005), TLT (2002), SPY (1993), SP500 stocks
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SIGNAL_WINDOW = 42      # IAU vs TLT 42d return comparison
MOMENTUM_WINDOW = 63    # SP500 stock 63d momentum
TREND_WINDOW = 200      # SP500 200d SMA bear gate
STOCK_TREND = 200       # per-stock 200d SMA filter
TOP_K = 15
EXPOSURE = 0.97
REBALANCE_EVERY = 10    # biweekly

_IAU = "IAU"
_TLT = "TLT"
_SPY = "SPY"


class IauTltInflationSp500(Strategy):
    """IAU/TLT inflation-regime signal routing SP500 momentum vs bond/gold blend."""

    def __init__(
        self,
        signal_window: int = SIGNAL_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        stock_trend: int = STOCK_TREND,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            signal_window=signal_window,
            momentum_window=momentum_window,
            trend_window=trend_window,
            stock_trend=stock_trend,
            top_k=top_k,
            exposure=exposure,
            rebalance_every=rebalance_every,
        )
        self.signal_window = int(signal_window)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.stock_trend = int(stock_trend)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window, self.signal_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
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
        spy_bull = spy_now > spy_sma

        # --- IAU vs TLT inflation-regime signal ---
        inflation_regime: bool | None = None
        try:
            iau_hist = ctx.history(_IAU)
            tlt_hist = ctx.history(_TLT)
            if (iau_hist is not None and tlt_hist is not None
                    and len(iau_hist) >= self.signal_window + 2
                    and len(tlt_hist) >= self.signal_window + 2):
                iau_c = iau_hist["close"].dropna()
                tlt_c = tlt_hist["close"].dropna()
                if len(iau_c) >= self.signal_window and len(tlt_c) >= self.signal_window:
                    iau_ret = float(iau_c.iloc[-1] / iau_c.iloc[-self.signal_window] - 1.0)
                    tlt_ret = float(tlt_c.iloc[-1] / tlt_c.iloc[-self.signal_window] - 1.0)
                    if np.isfinite(iau_ret) and np.isfinite(tlt_ret):
                        inflation_regime = iau_ret > tlt_ret
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market → TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif inflation_regime is False:
            # Deflationary / safety regime → TLT + IAU blend
            if _TLT in live:
                target[_TLT] = self.exposure * 0.618
            if _IAU in live:
                target[_IAU] = self.exposure * 0.382
        else:
            # Inflation regime or signal unavailable → SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IAU):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue

                    # 63d momentum
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(ret):
                        continue

                    # Per-stock 200d SMA filter
                    if len(col) < self.stock_trend:
                        continue
                    sma = float(col.iloc[-self.stock_trend:].mean())
                    curr = float(col.iloc[-1])
                    if curr <= sma:
                        continue

                    scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_w = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_w

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
    return sp500_tickers() + [_TLT, _IAU, _SPY]


NAME = "iau_tlt_inflation_sp500"
HYPOTHESIS = (
    "IAU vs TLT 42d return as inflation/deflation regime: gold outperforms bonds "
    "→ inflation regime → top-15 SP500 by 63d momentum above 200d SMA equal-weight; "
    "TLT outperforms gold → deflation/safety → TLT 62%+IAU 38%; "
    "SPY 200d bear gate → TLT; biweekly rebalance; gold/bond spread as novel regime gate"
)

UNIVERSE = _universe

STRATEGY = IauTltInflationSp500()
