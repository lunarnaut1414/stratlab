"""Zweig-style SP500 breadth-thrust signal — gen_7 opus-2 (gap_finder).

Hypothesis: A Zweig-style breadth thrust — fraction of SP500 stocks above
their own 50d SMA, then the 10d rate-of-change of that fraction — is a
distinct breadth-impulse signal not currently on the leaderboard. RSP-based
breadth (cap-weighted vs equal-weight ratio) is already mined; the explicit
count-based breadth ratio with momentum overlay isn't.

Logic:
  - Each rebalance compute fraction = (#stocks above own 50d SMA) / total.
  - 10d ROC of fraction = breadth thrust (positive = breadth improving).
  - thrust > 0 AND SPY > 200d -> SP500 top-15 by 63d momentum (risk-on).
  - thrust <= 0 OR SPY < 200d -> TLT 60% + SHY 37% (defensive).
  - Biweekly rebalance.

Distinction: count-based breadth-thrust impulse rather than the cap-weighted
RSP/SPY ratio used by existing breadth strategies. The 10d ROC catches
breadth inflection points rather than absolute breadth level.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
SMA_WINDOW = 50
THRUST_WINDOW = 10
STOCK_MOM_WINDOW = 63
TOP_K = 15
TREND_WINDOW = 200
EXPOSURE = 0.97


class ZweigBreadthThrustSp500(Strategy):
    """SP500 breadth-thrust impulse gates SP500 momentum vs TLT/SHY defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sma_window: int = SMA_WINDOW,
        thrust_window: int = THRUST_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sma_window=sma_window,
            thrust_window=thrust_window,
            stock_mom_window=stock_mom_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sma_window = int(sma_window)
        self.thrust_window = int(thrust_window)
        self.stock_mom_window = int(stock_mom_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_mom_window,
                     self.sma_window + self.thrust_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d gate
        bull_market = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    bull_market = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # Breadth thrust: fraction above own 50d SMA, take 10d ROC
        thrust_positive = True
        signal_ok = False
        need = self.sma_window + self.thrust_window + 5
        prices = ctx.closes_window(need)
        if len(prices) >= self.sma_window + self.thrust_window:
            try:
                # For each day in last (thrust_window+1) days, compute fraction above SMA
                fractions = []
                for offset in range(self.thrust_window + 1):
                    end_idx = len(prices) - offset
                    if end_idx < self.sma_window:
                        break
                    sub = prices.iloc[end_idx - self.sma_window:end_idx]
                    last_row = sub.iloc[-1]
                    sma = sub.mean()
                    above = 0
                    total = 0
                    for sym in sub.columns:
                        last_p = last_row[sym]
                        s = sma[sym]
                        if np.isfinite(last_p) and np.isfinite(s):
                            total += 1
                            if last_p > s:
                                above += 1
                    if total > 0:
                        fractions.append(above / total)
                if len(fractions) >= 2:
                    # fractions[0] = today, fractions[-1] = 10 days ago
                    today_frac = fractions[0]
                    past_frac = fractions[-1]
                    thrust = today_frac - past_frac
                    thrust_positive = (thrust > 0)
                    signal_ok = True
            except Exception:
                pass

        target: dict[str, float] = {}
        if not bull_market or not signal_ok or not thrust_positive:
            for sym, w in [("TLT", 0.60), ("SHY", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # SP500 top-K by 63d momentum
            need2 = self.stock_mom_window + 5
            prices2 = ctx.closes_window(need2)
            if len(prices2) < self.stock_mom_window:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices2.columns:
                    col = prices2[sym].dropna()
                    if len(col) < self.stock_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.stock_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret
                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
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
    return sp500_tickers() + ["SPY", "TLT", "SHY"]


NAME = "opus2_zweig_breadth_thrust_sp500"
HYPOTHESIS = (
    "Zweig-style SP500 breadth-thrust: fraction of SP500 stocks above own 50d SMA, take 10d ROC; "
    "positive thrust + SPY > 200d hold SP500 top-15 63d momentum; thrust negative or bear hold "
    "TLT 60%+SHY 37%; biweekly rebalance; count-based breadth-impulse novel vs RSP-ratio."
)
UNIVERSE = _universe

STRATEGY = ZweigBreadthThrustSp500()
