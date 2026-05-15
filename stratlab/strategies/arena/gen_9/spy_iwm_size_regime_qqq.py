"""SPY/IWM Size Factor Leadership Routing — gen_9 sonnet-8

Hypothesis:
Use SPY vs IWM 42d return spread as a size-factor regime indicator:
  - SPY outperforms IWM on 42d return (large-cap/mega-cap quality leadership):
    hold QQQ 97% — mega-cap tech outperformance dominates this regime
  - IWM outperforms SPY (small-cap breadth) AND SPY above 200d SMA:
    hold IWM 50% + SPY 47% — capture small-cap + broad equity blend
  - SPY below 200d SMA (bear regime): hold TLT 97%
Rebalance every 5 bars (weekly).

Rationale:
When large-cap quality leaders (mega-cap tech/growth) outperform small-caps,
it reflects a "quality-up" risk regime where investors prefer known growth
compounders. QQQ, heavily weighted toward mega-cap tech, thrives in this
environment. When small-caps lead large-caps, it signals broad economic
participation and reflation — a more balanced SPY+IWM blend captures the
breadth while maintaining equity exposure. This signal is inverted from
what might seem intuitive: SPY>IWM -> QQQ (not SPY), because mega-cap
leadership concentrates in the tech names that dominate QQQ.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 42           # 42d return spread SPY vs IWM
TREND_WINDOW = 200        # SPY 200d SMA bear gate
EXPOSURE = 0.97


class SpyIwmSizeRegimeQQQ(Strategy):
    """SPY/IWM 42d size regime routing QQQ/IWM+SPY/TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.trend_window = int(trend_window)
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
            # --- SPY vs IWM 42d return spread ---
            try:
                iwm_hist = ctx.history("IWM")
            except KeyError:
                return []
            if len(iwm_hist) < self.mom_window + 2 or len(spy_hist) < self.mom_window + 2:
                return []
            iwm_close = iwm_hist["close"].dropna()
            if len(iwm_close) < self.mom_window + 1 or len(spy_close) < self.mom_window + 1:
                return []

            spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-self.mom_window]) - 1.0
            iwm_ret = float(iwm_close.iloc[-1]) / float(iwm_close.iloc[-self.mom_window]) - 1.0

            if not np.isfinite(spy_ret) or not np.isfinite(iwm_ret):
                target = {"SPY": self.exposure}
            elif spy_ret > iwm_ret:
                # Mega-cap quality leadership: QQQ
                target = {"QQQ": self.exposure}
            else:
                # Small-cap breadth: IWM + SPY blend
                target = {}
                if "IWM" in live:
                    target["IWM"] = 0.50
                if "SPY" in live:
                    target["SPY"] = 0.47

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


NAME = "gen9_spy_iwm_size_regime_qqq"
HYPOTHESIS = (
    "SPY vs IWM 42d return spread as size regime gate: SPY leads IWM (mega-cap) "
    "-> QQQ 97%; IWM leads SPY (small-cap breadth) AND SPY bull -> IWM 50%+SPY 47%; "
    "SPY bear -> TLT 97%; weekly rebalance."
)

UNIVERSE = ["QQQ", "SPY", "IWM", "TLT"]

STRATEGY = SpyIwmSizeRegimeQQQ()
