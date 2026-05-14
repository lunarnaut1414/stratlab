"""IEF/SHY Duration Carry Regime — gen_8 sonnet-2

Hypothesis: Use the 10d realized return of IEF minus SHY (duration carry) as a
yield-curve regime signal. Positive carry (IEF outperforming SHY) indicates a
steep, supportive yield curve → hold QQQ 97%. Negative carry (SHY outperforming)
indicates flat/inverted curve → hold TLT 60%+SHY 37%. Neutral zone → SPY 97%.

Rationale:
- Yield curve steepness is directly reflected in the carry between intermediate
  (IEF) and short-term (SHY) Treasuries — when the curve is steep, IEF earns more
  duration premium and outperforms SHY.
- This is distinct from TNX/IRX yield level approaches (which use yield spread
  directly) because it uses ETF realized returns — includes both price return
  and coupon carry, smoothed naturally without requiring data from ^TNX signal.
- Also distinct from TLT/SHY approaches which are at the long end of the curve.
- IEF-SHY carry is a more stable signal than short-term rate spreads.

Thresholds: carry > +0.15% (10d) → QQQ; carry < 0% → TLT+SHY; else SPY
Rebalance: every 5 bars (weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
CARRY_WINDOW = 10         # 10 trading days for IEF vs SHY return comparison
TREND_WINDOW = 200        # SPY 200d SMA outer gate
EXPOSURE = 0.97
# Carry thresholds: cumulative 10d return difference (not annualized)
CARRY_POSITIVE_THRESH = 0.0015   # IEF must beat SHY by at least 15bps over 10d
CARRY_NEGATIVE_THRESH = 0.0      # if IEF is below SHY → inverted/flat
_IEF = "IEF"
_SHY = "SHY"
_QQQ = "QQQ"
_SPY = "SPY"
_TLT = "TLT"


class IEFSHYCarryRegime(Strategy):
    """IEF minus SHY 10d carry regime → QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        carry_window: int = CARRY_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
        carry_positive_thresh: float = CARRY_POSITIVE_THRESH,
        carry_negative_thresh: float = CARRY_NEGATIVE_THRESH,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            carry_window=carry_window,
            trend_window=trend_window,
            exposure=exposure,
            carry_positive_thresh=carry_positive_thresh,
            carry_negative_thresh=carry_negative_thresh,
        )
        self.rebalance_every = int(rebalance_every)
        self.carry_window = int(carry_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self.carry_positive_thresh = float(carry_positive_thresh)
        self.carry_negative_thresh = float(carry_negative_thresh)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.carry_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear regime: full TLT defensive
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute 10d returns for IEF and SHY
            need = self.carry_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.carry_window:
                return []

            ief_carry = None
            if _IEF in prices.columns:
                col = prices[_IEF].dropna()
                if len(col) >= self.carry_window + 1:
                    ief_carry = float(col.iloc[-1] / col.iloc[-self.carry_window] - 1.0)

            shy_carry = None
            if _SHY in prices.columns:
                col = prices[_SHY].dropna()
                if len(col) >= self.carry_window + 1:
                    shy_carry = float(col.iloc[-1] / col.iloc[-self.carry_window] - 1.0)

            # Determine regime
            if ief_carry is not None and shy_carry is not None:
                carry_spread = ief_carry - shy_carry
                if carry_spread >= self.carry_positive_thresh:
                    # Steep curve: QQQ (growth favored)
                    if _QQQ in live:
                        target[_QQQ] = self.exposure
                elif carry_spread <= self.carry_negative_thresh:
                    # Flat/inverted: TLT+SHY defensive
                    if _TLT in live:
                        target[_TLT] = self.exposure * 0.62
                    if _SHY in live:
                        target[_SHY] = self.exposure * 0.38
                else:
                    # Neutral carry: SPY
                    if _SPY in live:
                        target[_SPY] = self.exposure
            else:
                # Fallback: SPY when carry data unavailable
                if _SPY in live:
                    target[_SPY] = self.exposure

        orders: list[Order] = []

        # Exit positions not in target
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


UNIVERSE = [_IEF, _SHY, _QQQ, _SPY, _TLT]

NAME = "ief_shy_carry_regime"
HYPOTHESIS = (
    "IEF/SHY duration carry regime: 10d IEF return minus SHY return as curve signal; "
    "positive carry (>=15bps) hold QQQ 97%; negative carry (<0) hold TLT 60%+SHY 37%; "
    "neutral hold SPY 97%; bear (SPY < 200d SMA) hold TLT 97%; weekly rebalance; "
    "ETF-carry signal distinct from yield-level approaches"
)

STRATEGY = IEFSHYCarryRegime()
