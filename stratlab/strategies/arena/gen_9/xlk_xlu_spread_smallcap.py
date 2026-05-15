"""gen_9 sonnet-1 — XLK/XLU Growth vs Defensive Spread → Small-Cap Rotation

Hypothesis: Use the spread between XLK (tech/growth) and XLU (utilities/defensive)
42-day momentum as a risk-appetite barometer. When tech leads defensives (XLK 42d
return > XLU 42d return), the market is risk-on → hold IJR (S&P 600 small-cap ETF)
which benefits most from risk-appetite expansion. When defensives lead, route to
SPY (moderate risk-off). SPY 200d SMA outer bear gate routes fully to TLT.

Rationale:
- XLK vs XLU relative momentum is a clean growth vs defensive spread not used as
  primary signal in any prior round (gen5-8 leaderboard searched).
- Routes to IJR rather than individual stocks or QQQ — small-cap index has
  different daily return pattern than SP500 stock-selection strategies.
- 3-tier structure (IJR risk-on / SPY neutral / TLT bear) is simple and interpretable,
  avoiding overfitting of multi-threshold gates.
- Small-cap historically outperforms in risk-on regimes while having lower peak
  correlation to large-cap momentum strategies.

Coverage (all cover IS 2010-2018):
  IJR (2000), XLK (1998), XLU (1998), SPY (1993), TLT (2002), ^VIX-omitted
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 42        # 42d return for XLK vs XLU spread
SPY_TREND_WINDOW = 200      # 200d SMA outer bear gate
REBALANCE_EVERY = 10        # biweekly
EXPOSURE = 0.97

_XLK = "XLK"
_XLU = "XLU"
_IJR = "IJR"   # S&P 600 small-cap
_SPY = "SPY"
_TLT = "TLT"


class XlkXluSpreadSmallcap(Strategy):
    """XLK-XLU sector spread drives small-cap (IJR) vs SPY vs TLT allocation."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
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
            # Bear market → full TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute XLK and XLU 42d returns
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)

            xlk_ret: float | None = None
            xlu_ret: float | None = None

            if _XLK in prices.columns:
                xlk_col = prices[_XLK].dropna()
                if len(xlk_col) >= self.momentum_window:
                    xlk_ret = float(xlk_col.iloc[-1] / xlk_col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(xlk_ret):
                        xlk_ret = None

            if _XLU in prices.columns:
                xlu_col = prices[_XLU].dropna()
                if len(xlu_col) >= self.momentum_window:
                    xlu_ret = float(xlu_col.iloc[-1] / xlu_col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(xlu_ret):
                        xlu_ret = None

            if xlk_ret is None or xlu_ret is None:
                # Signal unavailable → SPY as default
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                spread = xlk_ret - xlu_ret  # positive = tech leads = risk-on

                if spread > 0.0:
                    # Tech leads → risk-on → small-cap IJR
                    if _IJR in live:
                        target[_IJR] = self.exposure
                    elif _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Defensives lead → moderate risk-off → SPY
                    if _SPY in live:
                        target[_SPY] = self.exposure

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


UNIVERSE = [_XLK, _XLU, _IJR, _SPY, _TLT]

NAME = "xlk_xlu_spread_smallcap"
HYPOTHESIS = (
    "XLK vs XLU 42d momentum spread as growth/defensive risk-appetite signal: "
    "tech leads (XLK > XLU) → IJR small-cap 97%; defensives lead → SPY 97%; "
    "SPY below 200d SMA → TLT 97%; biweekly rebalance; XLK-XLU spread routing to "
    "small-cap ETF untouched by prior rounds"
)

STRATEGY = XlkXluSpreadSmallcap()
