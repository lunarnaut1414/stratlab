"""XLY/XLP Consumer Sentiment Ratio as QQQ/SPY/TLT Regime Gate — gen_9 sonnet-8

Hypothesis:
Use XLY (consumer discretionary) vs XLP (consumer staples) 63d return spread
as a consumer-risk / economic cycle signal:
  - XLY leads XLP by > 2% over 63 days (strong risk appetite, cyclical spending):
    hold QQQ 97% (growth tilted, tech leadership)
  - XLP leads XLY OR spread -2% to +2% (neutral/defensive consumer preference)
    AND SPY above 200d SMA: hold SPY 60% + IEF 37%
  - SPY below 200d SMA (bear regime): hold TLT 97%

Rebalance weekly (5 bars) on signal change or scheduled.

Rationale:
XLY/XLP ratio is a classic consumer-cycle barometer: when households buy
discretionary goods (autos, luxury, retail, restaurants) over staples (food,
utilities, household products), economic confidence is high. This predicts
tech-led growth alpha in QQQ. The SPY trend filter catches the overall macro
bear. The spread is distinct from VIX level, credit spreads, yield-curve slope,
and all other signals on the leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 63           # 63d return spread XLY vs XLP
TREND_WINDOW = 200        # SPY 200d SMA gate
SPREAD_THRESHOLD = 0.02   # XLY must lead XLP by >2% to go full QQQ
EXPOSURE = 0.97


class XlyXlpConsumerRegimeQQQ(Strategy):
    """XLY/XLP consumer spread routing QQQ/SPY+IEF/TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        trend_window: int = TREND_WINDOW,
        spread_threshold: float = SPREAD_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            trend_window=trend_window,
            spread_threshold=spread_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.trend_window = int(trend_window)
        self.spread_threshold = float(spread_threshold)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(closes_now[s]) for s in closes_now.index
                if float(closes_now[s]) > 0}

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        if not spy_bull:
            target = {"TLT": self.exposure}
        else:
            # --- XLY vs XLP 63d spread ---
            try:
                xly_hist = ctx.history("XLY")
                xlp_hist = ctx.history("XLP")
            except KeyError:
                return []
            if len(xly_hist) < self.mom_window + 2 or len(xlp_hist) < self.mom_window + 2:
                return []
            xly_close = xly_hist["close"].dropna()
            xlp_close = xlp_hist["close"].dropna()
            if len(xly_close) < self.mom_window + 1 or len(xlp_close) < self.mom_window + 1:
                return []
            xly_ret = float(xly_close.iloc[-1]) / float(xly_close.iloc[-self.mom_window]) - 1.0
            xlp_ret = float(xlp_close.iloc[-1]) / float(xlp_close.iloc[-self.mom_window]) - 1.0
            spread = xly_ret - xlp_ret

            if spread > self.spread_threshold:
                # Risk-on consumer: QQQ
                target = {"QQQ": self.exposure}
            else:
                # Neutral/defensive: SPY + IEF blend
                target = {}
                if "SPY" in live:
                    target["SPY"] = 0.60
                if "IEF" in live:
                    target["IEF"] = 0.37

        # --- Build orders ---
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "gen9_xly_xlp_consumer_regime_qqq"
HYPOTHESIS = (
    "XLY vs XLP 63d return spread as consumer-sentiment regime gate: "
    "XLY leads XLP by >2% -> QQQ 97%; neutral/defensive spread AND SPY bull "
    "-> SPY 60%+IEF 37%; SPY below 200d SMA -> TLT 97%; weekly rebalance."
)

UNIVERSE = ["QQQ", "SPY", "TLT", "IEF", "XLY", "XLP"]

STRATEGY = XlyXlpConsumerRegimeQQQ()
