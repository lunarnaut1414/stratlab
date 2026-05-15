"""Dividend Income + Credit Rotation — gen_9 sonnet-8

Hypothesis:
Rank VIG/DVY (dividend ETFs with full IS coverage) by composite 42d momentum.
Use JNK-vs-TLT 30d return spread as credit-risk gauge:
  - When JNK 30d return > TLT 30d return (credit supportive, risk-on income):
    hold top-2 dividend ETFs by 42d momentum equally weighted.
  - When JNK 30d return <= TLT 30d return (credit stressed, TLT outperforms):
    rotate to TLT 60% + IEF 37% (safety in duration).
  - SPY 200d SMA outer bear gate: if SPY below 200d SMA, hold TLT 97%.
Rebalance every 21 bars (monthly) to limit turnover.

Rationale:
Dividend ETFs (VIG=quality growth dividend, DVY=high-yield dividend) represent
income-oriented equity. The JNK/TLT spread is a simple credit impulse: when
HY credit outperforms duration (JNK > TLT in total return), income-equity
is a risk-on mode. When TLT leads, capital is fleeing to safety — rotate to
bonds. This combination targets the "carry + credit" regime that is
structurally different from the cross-sectional momentum and VIX-level
strategies dominating the leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
MOM_WINDOW = 42            # 42d momentum for ranking dividend ETFs
CREDIT_WINDOW = 30         # JNK vs TLT 30d return spread
TREND_WINDOW = 200         # SPY 200d SMA bear gate
EXPOSURE = 0.97

DIVIDEND_ETFS = ["VIG", "DVY"]
DEFENSIVE_HEAVY = "TLT"
DEFENSIVE_LIGHT = "IEF"
CREDIT_SIGNAL = "JNK"
RATE_SIGNAL = "TLT"
TREND_SIGNAL = "SPY"

DEFENSIVE_HEAVY_W = 0.60
DEFENSIVE_LIGHT_W = 0.37


class DividendIncomeCreditRotation(Strategy):
    """Dividend ETF rotation gated by JNK/TLT credit impulse and SPY trend."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        credit_window: int = CREDIT_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            credit_window=credit_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.credit_window = int(credit_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_window, self.trend_window, self.credit_window) + 5
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
            spy_hist = ctx.history(TREND_SIGNAL)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        if not spy_bull:
            # Full defensive: TLT 97%
            target = {DEFENSIVE_HEAVY: self.exposure}
        else:
            # --- Credit impulse: JNK 30d return vs TLT 30d return ---
            try:
                jnk_hist = ctx.history(CREDIT_SIGNAL)
                tlt_hist = ctx.history(RATE_SIGNAL)
            except KeyError:
                return []
            if len(jnk_hist) < self.credit_window + 2 or len(tlt_hist) < self.credit_window + 2:
                return []
            jnk_close = jnk_hist["close"].dropna()
            tlt_close = tlt_hist["close"].dropna()
            if len(jnk_close) < self.credit_window + 1 or len(tlt_close) < self.credit_window + 1:
                return []
            jnk_ret = float(jnk_close.iloc[-1]) / float(jnk_close.iloc[-self.credit_window]) - 1.0
            tlt_ret = float(tlt_close.iloc[-1]) / float(tlt_close.iloc[-self.credit_window]) - 1.0

            credit_on = jnk_ret > tlt_ret

            if not credit_on:
                # Credit stressed: TLT/IEF defensive
                target: dict[str, float] = {}
                if DEFENSIVE_HEAVY in live:
                    target[DEFENSIVE_HEAVY] = DEFENSIVE_HEAVY_W
                if DEFENSIVE_LIGHT in live and DEFENSIVE_LIGHT in closes_now.index:
                    target[DEFENSIVE_LIGHT] = DEFENSIVE_LIGHT_W
            else:
                # Credit supportive: rank dividend ETFs by 42d momentum
                scores: dict[str, float] = {}
                need = self.mom_window + 2
                prices = ctx.closes_window(need)
                for sym in DIVIDEND_ETFS:
                    if sym not in prices.columns:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.mom_window])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) == 0:
                    target = {DEFENSIVE_HEAVY: self.exposure}
                else:
                    # Equal-weight the available dividend ETFs (up to top-2)
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:2]
                    w = self.exposure / len(ranked)
                    target = {sym: w for sym in ranked}

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


NAME = "gen9_dividend_income_credit_rotation"
HYPOTHESIS = (
    "Dividend income rotation: rank VIG/DVY by 42d momentum, hold top-2 when "
    "JNK 30d return > TLT 30d return (credit-on); rotate to TLT+IEF 60/37 when "
    "credit stressed; SPY 200d bear gate to TLT 97%; monthly rebalance."
)

UNIVERSE = ["VIG", "DVY", "JNK", "TLT", "IEF", "SPY"]

STRATEGY = DividendIncomeCreditRotation()
