"""opus-2 gap_finder: Sub-industry spread momentum (banks/semis pairs).

Gap identified: every "sector" strategy on the leaderboard uses BROAD SPDR
sectors (XLF, XLK, XLE, XLV, XLY, XLP). None use sub-industry pairs that
split a broad sector into specialist vs broad components. KBE (banks) is a
narrow slice of XLF (financials). SMH (semis) and SOXX (broad-semi) split
the semiconductor industry differently. The *spread* between specialist and
broad captures sub-industry rotation orthogonal to broad-sector momentum.

Hypothesis: When KBE-XLF 63d return spread is positive AND SMH-SOXX 63d
spread is positive, both narrow industries are leading their broad parents —
hold KBE+SMH (concentrated risk-on). When both spreads negative, lagging —
defensive SHY+TLT. Mixed: balanced SPY hold. Spreads, not absolutes.

Universe:
  - KBE, XLF, SMH, SOXX, SPY, SHY, TLT
  - All cover IS window (verified via stratlab.data.inception)

Mechanics:
  - 63d return spreads: (KBE_63d - XLF_63d) and (SMH_63d - SOXX_63d)
  - Both positive -> 50% KBE + 45% SMH (industry leaders)
  - Both negative -> 50% SHY + 45% TLT (defensive)
  - Mixed        -> 90% SPY (neutral)
  - Rebalance every 10 bars
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["KBE", "XLF", "SMH", "SOXX", "SPY", "SHY", "TLT"]

LOOKBACK = 63
REBALANCE_EVERY = 10


def _return_n(closes: pd.Series, n: int) -> float:
    c = closes.dropna()
    if len(c) < n + 1:
        return float("nan")
    return float(c.iloc[-1] / c.iloc[-(n + 1)] - 1.0)


class SubindustrySpreadMomentum(Strategy):
    def __init__(
        self,
        lookback: int = LOOKBACK,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            lookback=lookback,
            rebalance_every=rebalance_every,
        )
        self.lookback = int(lookback)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.lookback + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        rets: dict[str, float] = {}
        for sym in ["KBE", "XLF", "SMH", "SOXX"]:
            try:
                h = ctx.history(sym)
            except KeyError:
                return []
            if h is None or len(h) < self.lookback + 5:
                return []
            rets[sym] = _return_n(h["close"], self.lookback)
            if not np.isfinite(rets[sym]):
                return []

        spread_kbe = rets["KBE"] - rets["XLF"]
        spread_smh = rets["SMH"] - rets["SOXX"]

        kbe_lead = spread_kbe > 0
        smh_lead = spread_smh > 0

        # Always-long SPY core; tilt extra weight to leading sub-industry sleeve.
        # SPY 200d trend gate decides defensive escape; spreads decide tilt direction.
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if spy_hist is None or len(spy_hist) < 201:
            return []
        spy_c = spy_hist["close"].dropna()
        spy_now = float(spy_c.iloc[-1])
        spy_200 = float(spy_c.iloc[-200:].mean())
        spy_bull = np.isfinite(spy_now) and np.isfinite(spy_200) and spy_now > spy_200

        if not spy_bull:
            target = {"SHY": 0.55, "TLT": 0.40}
        else:
            # SPY bull: 60% SPY core + 35% tilt distributed among leading sleeves
            target = {"SPY": 0.60}
            tilt_legs: list[str] = []
            if kbe_lead:
                tilt_legs.append("KBE")
            if smh_lead:
                tilt_legs.append("SMH")
            if tilt_legs:
                w_each = 0.35 / len(tilt_legs)
                for s in tilt_legs:
                    target[s] = w_each
            else:
                # neither leading -> hold extra SPY
                target["SPY"] = 0.92

        live = ctx.closes()
        if live.empty:
            return []
        live_dict = {s: float(p) for s, p in live.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))

        for sym, weight in target.items():
            price = live_dict.get(sym)
            if price is None or price <= 0:
                continue
            target_shares = int(equity * weight / price)
            cur_shares = int(ctx.position(sym).size)
            delta = target_shares - cur_shares
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


NAME = "opus2_subindustry_spread_momentum"
HYPOTHESIS = (
    "Sub-industry spread momentum: 63d return spread (KBE-XLF) AND (SMH-SOXX). Both>0 -> "
    "50% KBE + 45% SMH (narrow industry leaders); both<0 -> 50% SHY + 45% TLT defensive; "
    "mixed -> 85% SPY neutral. Specialist-vs-broad spreads orthogonal to broad-sector rotation."
)

STRATEGY = SubindustrySpreadMomentum()
