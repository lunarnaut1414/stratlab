"""RSP/SPY Equal-Weight Breadth Regime Gate for QQQ/SPY+IEF/TLT — gen_9 sonnet-8

Hypothesis:
Use RSP (equal-weight S&P500 ETF) vs SPY (cap-weight S&P500) 42d return
spread as a market-breadth barometer:
  - RSP outperforms SPY on 42d return (broad participation, risk-on):
    hold QQQ 97% (growth/tech leading broadly, capture tech premium)
  - SPY outperforms RSP (narrow mega-cap leadership) AND SPY above 200d SMA:
    hold SPY 60% + IEF 37% (moderate risk-on but not full QQQ)
  - SPY below 200d SMA (bear regime): hold TLT 97%
Rebalance every 5 bars (weekly).

Rationale:
When equal-weight S&P500 outperforms cap-weight, it signals that a broad
cross-section of stocks is participating in the rally (not just a handful of
mega-caps). This is a healthy bull signal — historically, broad market
participation correlates with stronger subsequent returns. QQQ captures the
growth premium in that environment. When only mega-caps lead (SPY beats RSP),
the rally is narrow; a more cautious SPY+bond blend is appropriate. This
signal is distinct from: VIX levels, credit spreads, yield curve slope,
consumer sentiment (XLY/XLP), and sector Sharpe competition.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 42           # 42d relative return RSP vs SPY
TREND_WINDOW = 200        # SPY 200d SMA bear gate
ABS_MOM_THRESHOLD = 0.0   # RSP must have positive absolute 42d return to go QQQ
EXPOSURE = 0.97


class RspSpyBreadthQqqRouting(Strategy):
    """RSP/SPY 42d breadth spread routing QQQ/SPY+IEF/TLT."""

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
        self.abs_mom_threshold = float(ABS_MOM_THRESHOLD)
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
            # --- RSP vs SPY 42d return spread ---
            try:
                rsp_hist = ctx.history("RSP")
            except KeyError:
                return []
            if len(rsp_hist) < self.mom_window + 2 or len(spy_hist) < self.mom_window + 2:
                return []
            rsp_close = rsp_hist["close"].dropna()
            if len(rsp_close) < self.mom_window + 1 or len(spy_close) < self.mom_window + 1:
                return []

            rsp_ret = float(rsp_close.iloc[-1]) / float(rsp_close.iloc[-self.mom_window]) - 1.0
            spy_ret_42 = float(spy_close.iloc[-1]) / float(spy_close.iloc[-self.mom_window]) - 1.0

            if not np.isfinite(rsp_ret) or not np.isfinite(spy_ret_42):
                target = {"SPY": 0.60, "IEF": 0.37}
            elif rsp_ret > spy_ret_42 and rsp_ret > self.abs_mom_threshold:
                # Broad participation with positive absolute momentum: QQQ
                target = {"QQQ": self.exposure}
            else:
                # Narrow mega-cap leadership: SPY + IEF blend
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


NAME = "gen9_rsp_spy_breadth_qqq_routing"
HYPOTHESIS = (
    "RSP vs SPY 42d return spread as breadth regime gate: RSP outperforms SPY -> "
    "QQQ 97%; SPY leads RSP AND SPY bull -> SPY 60%+IEF 37%; SPY bear -> TLT 97%; "
    "weekly rebalance; breadth-routing to QQQ not used in prior rounds."
)

UNIVERSE = ["QQQ", "SPY", "RSP", "TLT", "IEF"]

STRATEGY = RspSpyBreadthQqqRouting()
