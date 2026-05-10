"""High-yield credit momentum to QQQ rotation.

Hypothesis:
  JNK (high yield) is a risk-on asset that tends to lead equity markets.
  When JNK is in an uptrend (above its 30d SMA), buy QQQ (Nasdaq-100,
  high-beta equity). When JNK breaks below its 30d SMA, rotate to TLT
  (duration + safety). Rebalance weekly (5 bars).

  Secondary confirmation: require SPY to also be above its 100d SMA for
  QQQ allocation (double confirmation prevents false signals during
  sideways JNK with falling equities).

Rationale:
  High-yield credit spreads widen before equity selloffs and tighten before
  equity rallies. JNK price directly captures this: falling JNK price =
  widening spreads = credit stress ahead of equities. This signal is
  leading, not contemporaneous.

  Using QQQ (not SPY) amplifies the return capture during risk-on regimes.
  The 30d MA is shorter than typical 50d to be more responsive to credit
  turning points.

  Unlike gen5_credit_spread_hyg_lqd (JNK/LQD ratio MA crossover → JNK or LQD),
  this strategy ROUTES TO EQUITIES (QQQ) on credit strength, not just
  between credit instruments. This produces a different return profile.

Diversification vs leaderboard:
  - gen5_credit_spread_hyg_lqd: JNK/LQD ratio signal → stays in bonds.
    This routes to QQQ when JNK uptrend → much more equity-like in bull markets.
  - All SP500 momentum strategies: signal is credit (JNK), not price momentum.
  - gen5_cyclicals_vs_defensives: sector-based regime, this is credit-based.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 30          # JNK short-term trend
SPY_MA = 100         # SPY secondary confirmation
REBALANCE_EVERY = 5  # weekly
EXPOSURE = 0.97

_QQQ = "QQQ"
_TLT = "TLT"
_JNK = "JNK"
_SPY = "SPY"


class HyCreditQqqRotation(Strategy):
    """QQQ when JNK uptrend + SPY above 100d SMA; TLT otherwise."""

    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        spy_ma: int = SPY_MA,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            spy_ma=spy_ma,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.spy_ma = int(spy_ma)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.spy_ma) + 10
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

        # --- JNK trend ---
        jnk_bullish = False
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 1:
                jnk_close = jnk_hist["close"].dropna()
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_bullish = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        # --- SPY 100d SMA confirmation ---
        spy_bull = False
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.spy_ma + 1:
                spy_close = spy_hist["close"].dropna()
                spy_sma = float(spy_close.iloc[-self.spy_ma:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if jnk_bullish and spy_bull:
            # Risk-on: credit uptrend + equity uptrend → QQQ
            if _QQQ in closes_now.index:
                target[_QQQ] = self.exposure
        else:
            # Risk-off: either credit weakening or equity downtrend → TLT
            if _TLT in closes_now.index:
                target[_TLT] = self.exposure

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


NAME = "hy_credit_qqq_rotation"
HYPOTHESIS = (
    "High-yield credit momentum to QQQ rotation: hold QQQ 97% when JNK above "
    "30d SMA AND SPY above 100d SMA (dual credit+trend confirmation); TLT 97% "
    "otherwise; weekly rebalance. Routes to equities on credit strength."
)

UNIVERSE = ["QQQ", "TLT", "JNK", "SPY"]

STRATEGY = HyCreditQqqRotation()
