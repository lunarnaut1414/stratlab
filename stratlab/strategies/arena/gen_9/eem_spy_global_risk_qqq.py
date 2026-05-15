"""EEM vs SPY Global Risk-Appetite Regime Gate — gen_9 sonnet-8

Hypothesis:
Use EEM (Emerging Markets ETF) vs SPY (US large-cap) 42d return spread as a
global risk-appetite and dollar-cycle barometer:
  - EEM outperforms SPY on 42d return (global risk-on, dollar weakness, EM growth):
    hold QQQ 97% (tech/growth premium in global risk-on environment)
  - SPY outperforms EEM (US-domestic defensiveness, dollar strength) AND SPY
    above 200d SMA: hold SPY 60% + IEF 37% (moderate risk-on)
  - SPY below 200d SMA (bear regime): hold TLT 97%
Rebalance every 5 bars (weekly).

Rationale:
When EM outperforms US equities, it signals: (1) global growth accelerating,
(2) dollar weakening (favorable for risk assets), (3) risk appetite high across
geographies. Tech/growth (QQQ) benefits disproportionately in this environment.
When US outperforms EM, it signals: dollar strength, risk-off globally, but
domestic US resilience — hold SPY with bond buffer. The EEM/SPY spread is a
genuine global-macro signal distinct from: VIX level, credit spreads (JNK),
yield curve slope, consumer sentiment (XLY/XLP), and sector Sharpe competition.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 42           # 42d return spread EEM vs SPY
TREND_WINDOW = 200        # SPY 200d SMA bear gate
EXPOSURE = 0.97


class EemSpyGlobalRiskQQQ(Strategy):
    """EEM/SPY 42d return spread routing QQQ/SPY+IEF/TLT."""

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
            # --- EEM vs SPY 42d return spread ---
            try:
                eem_hist = ctx.history("EEM")
            except KeyError:
                return []
            if len(eem_hist) < self.mom_window + 2 or len(spy_hist) < self.mom_window + 2:
                return []
            eem_close = eem_hist["close"].dropna()
            if len(eem_close) < self.mom_window + 1 or len(spy_close) < self.mom_window + 1:
                return []

            eem_ret = float(eem_close.iloc[-1]) / float(eem_close.iloc[-self.mom_window]) - 1.0
            spy_ret_42 = float(spy_close.iloc[-1]) / float(spy_close.iloc[-self.mom_window]) - 1.0

            if not np.isfinite(eem_ret) or not np.isfinite(spy_ret_42):
                target = {"SPY": 0.60, "IEF": 0.37}
            elif eem_ret > spy_ret_42:
                # Global risk-on: QQQ
                target = {"QQQ": self.exposure}
            else:
                # US-defensive regime: SPY + IEF blend
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


NAME = "gen9_eem_spy_global_risk_qqq"
HYPOTHESIS = (
    "EEM vs SPY 42d return spread as global risk-appetite gate: EEM leads SPY "
    "-> QQQ 97%; SPY leads EEM AND SPY bull -> SPY 60%+IEF 37%; SPY bear "
    "-> TLT 97%; weekly rebalance."
)

UNIVERSE = ["QQQ", "SPY", "EEM", "TLT", "IEF"]

STRATEGY = EemSpyGlobalRiskQQQ()
