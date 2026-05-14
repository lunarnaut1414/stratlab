"""TNX Yield-Direction Rate-Regime SP500 Momentum — gen_8 sonnet-5

Hypothesis: Use the 10-year Treasury yield (^TNX) direction — specifically
whether its 20d MA is above or below its 60d MA — as a macro regime gate
for SP500 stock momentum.

- Falling rates (TNX 20d MA < TNX 60d MA): growth-supportive environment.
  Hold top-20 SP500 stocks by 126d momentum above their 200d SMA.
  Equal-weight. These are momentum stocks that benefit from lower discount rates.

- Rising rates (TNX 20d MA > TNX 60d MA): duration/growth headwind.
  Rotate to IEF 60% + GLD 37% (mid-duration treasuries + gold as real-asset
  hedge against rate volatility). This avoids the rate-sensitive growth stock
  drawdown that accompanies rate-rising regimes.

- Outer bear gate: SPY below 200d SMA → TLT 60% + GLD 37%.

Rationale: Rising/falling 10Y yield direction is one of the most important
macro factors for growth equity. The 20d/60d MA crossover on TNX gives a
smoother, less whippy signal than level thresholds. The IS window 2010-2018
includes the ZRP era (2010-2015 rates falling/flat) and two tightening cycles
(2013 taper, 2015-2018 hike) — providing both regimes. Distinct from all
existing VIX/JNK/credit/VIX-percentile gates.

Rebalance: weekly (5 bars). Trades enough to easily clear 50-trade minimum.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
MOMENTUM_WINDOW = 126      # ~6 months
TREND_WINDOW = 200         # SPY 200d SMA
TNX_FAST_MA = 20           # TNX 20d MA
TNX_SLOW_MA = 60           # TNX 60d MA
TOP_K = 20                 # top-20 stocks
EXPOSURE = 0.97
_SPY = "SPY"
_TNX = "^TNX"
_TLT = "TLT"
_IEF = "IEF"
_GLD = "GLD"


class TnxRateRegimeSP500Momentum(Strategy):
    """TNX 20d/60d MA crossover gates SP500 momentum vs IEF+GLD defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        tnx_fast_ma: int = TNX_FAST_MA,
        tnx_slow_ma: int = TNX_SLOW_MA,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            tnx_fast_ma=tnx_fast_ma,
            tnx_slow_ma=tnx_slow_ma,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.tnx_fast_ma = int(tnx_fast_ma)
        self.tnx_slow_ma = int(tnx_slow_ma)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.tnx_slow_ma, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend gate ---
        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT 60% + GLD 37%
            for sym, w in [(_TLT, 0.60), (_GLD, 0.37)]:
                if sym in live:
                    target[sym] = w * self.exposure
        else:
            # Check TNX regime: 20d MA vs 60d MA
            tnx_rising = False  # default assumption: falling rates
            try:
                tnx_hist = ctx.history(_TNX)
                if tnx_hist is not None and len(tnx_hist) >= self.tnx_slow_ma + 5:
                    tnx_close = tnx_hist["close"].dropna()
                    if len(tnx_close) >= self.tnx_slow_ma:
                        tnx_fast = float(tnx_close.iloc[-self.tnx_fast_ma:].mean())
                        tnx_slow = float(tnx_close.iloc[-self.tnx_slow_ma:].mean())
                        tnx_rising = tnx_fast > tnx_slow
            except Exception:
                pass

            if tnx_rising:
                # Rising rates: rotate to IEF 60% + GLD 37%
                for sym, w in [(_IEF, 0.60), (_GLD, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                # Falling rates: hold top-K SP500 momentum stocks
                prices = ctx.closes_window(self.momentum_window + 10)
                if len(prices) < self.momentum_window:
                    return []

                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _GLD):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    # Not enough candidates — fall to IEF defensive
                    if _IEF in live:
                        target[_IEF] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                    # Apply individual 200d SMA trend filter
                    selected: list[str] = []
                    for sym in ranked:
                        if len(selected) >= self.top_k:
                            break
                        sh = ctx.history(sym)
                        if len(sh) < self.trend_window:
                            continue
                        sc = sh["close"].dropna()
                        if len(sc) < self.trend_window:
                            continue
                        sma = float(sc.iloc[-self.trend_window:].mean())
                        price = live.get(sym, 0.0)
                        if price > sma:
                            selected.append(sym)

                    if not selected:
                        # All momentum stocks in downtrend — hold IEF
                        if _IEF in live:
                            target[_IEF] = self.exposure
                    else:
                        per_w = self.exposure / len(selected)
                        for sym in selected:
                            if sym in live:
                                target[sym] = per_w

        # --- Execute ---
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
    return sp500_tickers() + [_SPY, _TLT, _IEF, _GLD, _TNX]


NAME = "tnx_rate_regime_sp500_momentum"
HYPOTHESIS = (
    "TNX yield-direction rate-regime switcher: hold top-20 SP500 stocks by 126d momentum "
    "above 200d SMA when TNX 20d MA is BELOW TNX 60d MA (falling rates, growth-supportive); "
    "rotate to IEF 60%+GLD 37% when TNX rising (20d MA > 60d MA); "
    "weekly rebalance; avoids rising-rate headwinds on growth stocks"
)

UNIVERSE = _universe

STRATEGY = TnxRateRegimeSP500Momentum()
