"""Mid-cap and small-cap ETF momentum rotation with SPY 200d gate.

Hypothesis: The leaderboard contains many SPY/QQQ/IWM-based strategies but
none use MDY (mid-cap 400) or IJH (mid-cap 400) or IJR (small-cap 600)
explicitly. Mid-caps had a strong 2010-2018 run with different timing from
SPY (small-cap leadership episodes 2013, 2016) and a different drawdown
path (2011 small-cap crisis, 2015-2016 small-cap rout).

Strategy:
  - SPY 200d SMA gate: when SPY < 200d, hold IEF (defensive bond).
  - When SPY > 200d, rank MDY / IJH / IJR / SPY by 63d total return.
  - Hold top-2 equally weighted at 0.485 * EXPOSURE each (~94% gross).
  - Biweekly rebalance (every 10 bars).

Why this fills a gap:
  - Phase 2 brief: "Mid-cap-only (IJR/IJH) momentum — leaderboard has
    SPY/QQQ/IWM but no MDY/IJH".
  - Saturated themes are intra-equity *sector* and *international/EM*
    rotation. Mid-cap and small-cap are *size-factor* rotation, distinct
    cluster. None of the 26 gen_6 leaderboard rows uses MDY/IJH/IJR.
  - smallcap_leadership_rotation uses IWM only; this strategy uses MDY+IJH+IJR
    so it's a different signal carrier.

Universe is tiny (5 ETFs + 1 index) so backtest is fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["MDY", "IJH", "IJR", "SPY", "IEF"]

CANDIDATES = ["MDY", "IJH", "IJR", "SPY"]
TOP_K = 2
MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
REBALANCE_EVERY = 10
EXPOSURE = 0.97


class MidcapMomentumRotation(Strategy):
    def __init__(
        self,
        top_k: int = TOP_K,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            top_k=top_k,
            momentum_window=momentum_window,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.top_k = int(top_k)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend gate
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 1:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window + 1:
                    spy_now = float(spy_close.iloc[-1])
                    spy_ma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = spy_now > spy_ma
        except KeyError:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Defensive
            if "IEF" in live:
                target["IEF"] = self.exposure
        else:
            # Score candidates by momentum_window total return
            scores: dict[str, float] = {}
            prices_window = ctx.closes_window(self.momentum_window + 5)
            if len(prices_window) < self.momentum_window:
                return []
            for sym in CANDIDATES:
                if sym not in prices_window.columns:
                    continue
                col = prices_window[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) >= self.top_k:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                longs = ranked[: self.top_k]
                per_weight = self.exposure / len(longs)
                for sym in longs:
                    if sym in live:
                        target[sym] = per_weight

            if not target and "SPY" in live:
                target["SPY"] = self.exposure

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "midcap_momentum_rotation"
HYPOTHESIS = (
    "Mid/small-cap momentum rotation MDY/IJH/IJR/SPY top-2 by 63d return when "
    "SPY>200d SMA; rotate to IEF when SPY<200d SMA; biweekly rebalance; "
    "size-factor rotation distinct from sector/international rotation themes."
)

STRATEGY = MidcapMomentumRotation()
