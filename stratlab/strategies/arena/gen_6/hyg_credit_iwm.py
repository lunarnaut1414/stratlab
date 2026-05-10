"""HYG credit gate → IWM small-cap rotation — gen_6 sonnet-7

Hypothesis: Small-cap stocks (IWM) benefit most from credit easing
because smaller companies rely more heavily on debt financing. Use
HYG above its 60d SMA as credit health signal AND SPY above 100d SMA
as trend gate. When both conditions are met: hold IWM 97%. When only
SPY is bullish (credit neutral): hold SPY 60% + IWM 37%. When SPY is
bearish: hold IEF 97%. Rebalance every 5 bars.

Rationale:
  Small-cap stocks have higher leverage ratios than large caps. Credit
  tightening disproportionately hurts small companies (higher refinancing
  costs, tighter bank lending). HYG (high-yield bond ETF) captures
  credit health across the entire corporate universe. When HYG trends up
  AND SPY is in an uptrend, it's a green light for small-cap outperformance.

  Distinct from existing leaderboard:
  - IWM (small-cap) as primary equity vehicle (not SPY/QQQ)
  - HYG 60d SMA (not JNK 20d or 30d) as credit gate
  - SPY/IWM blend in "partial credit" state (not binary all-or-nothing)
  - IEF as defensive (7-10yr, less volatile than TLT 20yr)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
HYG_MA = 60             # HYG 60d SMA for credit gate
TREND_WINDOW = 100      # SPY 100d SMA for equity trend
EXPOSURE = 0.97


class HYGCreditIWM(Strategy):
    """HYG credit gate for IWM small-cap rotation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        hyg_ma: int = HYG_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            hyg_ma=hyg_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.hyg_ma = int(hyg_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.hyg_ma, self.trend_window) + 5
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

        # SPY trend gate (100d SMA)
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # HYG credit health (60d SMA)
        hyg_healthy = True  # assume healthy if no data
        try:
            hyg_hist = ctx.history("HYG")
            if len(hyg_hist) >= self.hyg_ma + 2:
                hyg_close = hyg_hist["close"].dropna().values
                hyg_sma = float(np.mean(hyg_close[-self.hyg_ma:]))
                hyg_healthy = float(hyg_close[-1]) > hyg_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: IEF (intermediate duration)
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        elif hyg_healthy:
            # Credit healthy + bull market: full IWM (small-cap)
            if "IWM" in closes_now.index:
                target["IWM"] = self.exposure
        else:
            # Credit weakening but SPY still in trend: SPY + IWM blend
            if "SPY" in closes_now.index:
                target["SPY"] = 0.60 * self.exposure
            if "IWM" in closes_now.index:
                target["IWM"] = 0.37 * self.exposure

        # Build orders
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


NAME = "hyg_credit_iwm"
HYPOTHESIS = (
    "HYG credit gate IWM rotation: HYG>60d SMA AND SPY>100d SMA → IWM 97% (small-cap outperform); "
    "HYG weak but SPY bull → SPY 60%+IWM 37%; SPY bear → IEF 97%; weekly rebalance; "
    "credit-gated small-cap timing distinct from JNK→QQQ and credit→SP500 strategies"
)
UNIVERSE = ["IWM", "SPY", "IEF", "HYG"]
STRATEGY = HYGCreditIWM()
