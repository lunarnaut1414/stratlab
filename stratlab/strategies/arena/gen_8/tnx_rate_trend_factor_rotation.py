"""TNX Rate-Trend Factor ETF Rotation — gen_8 sonnet-6

Hypothesis: Use the 10yr Treasury yield (TNX) 20d momentum as a rate-trend
signal to rotate between rate-sensitive equity factor ETFs. Rising rates
(TNX 20d return > +0.5%): hold XLF (financials benefit) + MTUM (momentum);
Falling rates (TNX 20d return < -0.5%): hold XLRE (REITs benefit) + TLT;
Neutral: SPY + IEF blend. Weekly rebalance.

Rationale:
- TNX as SIGNAL (not traded): treasury yield trend drives sector rotations
  via financial vs defensive sector effects.
- XLF benefits from rising rates (net interest margin expansion).
- XLRE/Utilities suffer from rising rates and benefit from falling rates.
- MTUM added in rising-rate regime as markets in this phase often see growth
  stocks re-accelerating after rate normalization (2013-2015 pattern).
- Falling rates: REITs (XLRE) as yield alternative + TLT for duration.
- Neutral rate environment: SPY 60% + IEF 37% (balanced).

This uses TNX as signal without ever holding it. Rate-driven sector rotation
through ETFs is distinct from all existing strategies:
- Existing yield curve strategies (gen_6, gen_7) use 10Y-2Y SLOPE
- This uses 10Y TNX ABSOLUTE MOMENTUM (trend, not slope)
- Defensives in falling-rate regime via XLRE (not pure bond) — novel
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # weekly
TNX_MOM_WINDOW = 20      # 20d TNX yield momentum
RATE_RISE_THRESH = 0.005  # TNX 20d return > +0.5pp = rising rates
RATE_FALL_THRESH = -0.005 # TNX 20d return < -0.5pp = falling rates
EXPOSURE = 0.97

_TNX = "^TNX"   # 10yr yield — signal only
_XLF = "XLF"   # financials — benefit from rising rates
_MTUM = "MTUM"  # momentum factor — often in rising-rate equity expansions
_XLRE = "XLRE"  # REITs — benefit from falling rates (yield alternative)
_TLT = "TLT"   # long-duration bonds — benefit from falling rates
_SPY = "SPY"   # neutral market exposure
_IEF = "IEF"   # medium-duration bonds — neutral blend


class TnxRateTrendFactorRotation(Strategy):
    """TNX 20d momentum drives rotation between financial/momentum vs REIT/bond ETFs."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        tnx_mom_window: int = TNX_MOM_WINDOW,
        rate_rise_thresh: float = RATE_RISE_THRESH,
        rate_fall_thresh: float = RATE_FALL_THRESH,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            tnx_mom_window=tnx_mom_window,
            rate_rise_thresh=rate_rise_thresh,
            rate_fall_thresh=rate_fall_thresh,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.tnx_mom_window = int(tnx_mom_window)
        self.rate_rise_thresh = float(rate_rise_thresh)
        self.rate_fall_thresh = float(rate_fall_thresh)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.tnx_mom_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get TNX (10yr yield) as rate-trend signal
        try:
            tnx_hist = ctx.history(_TNX)
        except KeyError:
            return []
        if tnx_hist is None or len(tnx_hist) < self.tnx_mom_window + 2:
            return []

        tnx_close = tnx_hist["close"].dropna()
        if len(tnx_close) < self.tnx_mom_window + 1:
            return []

        # TNX 20d momentum: absolute change in yield level
        # TNX is quoted as %, so 4.5 means 4.5% yield
        tnx_now = float(tnx_close.iloc[-1])
        tnx_past = float(tnx_close.iloc[-self.tnx_mom_window])
        # Rate momentum as absolute change (in percentage points)
        rate_change = (tnx_now - tnx_past) / 100.0  # convert pp to decimal

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine rate regime and target allocations
        target: dict[str, float] = {}

        if rate_change > self.rate_rise_thresh:
            # Rising rates: financials + momentum factor
            # Split 50/50 between XLF and MTUM (if available)
            available = [s for s in [_XLF, _MTUM] if s in live]
            if not available:
                available = [s for s in [_SPY] if s in live]
            if available:
                per_slot = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_slot
        elif rate_change < self.rate_fall_thresh:
            # Falling rates: REITs + long-duration bonds
            available = [s for s in [_XLRE, _TLT] if s in live]
            if not available:
                available = [s for s in [_TLT, _IEF] if s in live]
            if available:
                per_slot = self.exposure / len(available)
                for sym in available:
                    target[sym] = per_slot
        else:
            # Neutral rate environment: balanced SPY + IEF
            available = [s for s in [_SPY, _IEF] if s in live]
            if available:
                if len(available) == 2:
                    target[_SPY] = 0.60
                    target[_IEF] = self.exposure - 0.60
                else:
                    target[available[0]] = self.exposure

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


UNIVERSE = [_XLF, _MTUM, _XLRE, _TLT, _SPY, _IEF, _TNX]

NAME = "tnx_rate_trend_factor_rotation"
HYPOTHESIS = (
    "TNX 10yr yield 20d momentum as rate-trend signal for factor ETF rotation: "
    "rising rates (TNX 20d change > +0.5pp) hold XLF+MTUM (financials + momentum); "
    "falling rates (< -0.5pp) hold XLRE+TLT (REITs + duration); "
    "neutral hold SPY+IEF blend; weekly rebalance; "
    "rate TREND (not slope) driving factor rotation distinct from existing yield-curve strategies"
)

STRATEGY = TnxRateTrendFactorRotation()
