"""KBE/KRE vs XLF Sub-Industry Spread — gen_8 opus-2 (gap_finder)

Hypothesis: Use the relative 63d momentum of regional+mid-cap banks
(KBE bank SPDR + KRE regional banks SPDR) versus broad financials
(XLF SPDR) as a credit-cycle-stage signal:

- When KBE+KRE composite outperforms XLF by >2% on 63d return: regional
  banks are leading, which historically coincides with credit-cycle
  EXPANSION (banks lending more, NIM expanding, regionals win on
  loan growth). Concentrate in KBE+KRE 50/50.
- When XLF leads KBE+KRE by >2%: large diversified financials lead
  (Berkshire, JPM, GS, large banks/insurance/asset managers) — usually
  late-cycle defensive financial regime. Hold XLF.
- Middle (neutral): hold XLF.
- SPY 200d SMA bear gate: full TLT.

Why this fills a gap (after 4 rounds):
- NO prior strategy uses SUB-INDUSTRY WITHIN a sector. All sector
  strategies use SPDR XL* macro sectors (XLK/XLV/XLF/XLI/...) as
  competing equal-tier ETFs. This goes a level deeper: it competes
  KBE (banks specifically) and KRE (regional banks specifically)
  against the broader XLF (banks + insurance + asset managers + REITs).
- Cache: KBE from 2005-11, KRE from 2006-06, XLF from 1998 — full IS
  window coverage.

Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
SPY_TREND_WINDOW = 200
SPREAD_THRESHOLD = 0.02
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_KBE = "KBE"
_KRE = "KRE"
_XLF = "XLF"


class KbeKreXlfSubIndustry(Strategy):
    """KBE+KRE vs XLF sub-industry spread within financials."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        spread_threshold: float = SPREAD_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            spread_threshold=spread_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.spread_threshold = float(spread_threshold)
        self.exposure = float(exposure)

    def _mom_return(self, ctx: BarContext, sym: str) -> float | None:
        try:
            h = ctx.history(sym)
        except KeyError:
            return None
        if h is None:
            return None
        cl = h["close"].dropna()
        if len(cl) < self.momentum_window + 1:
            return None
        try:
            r = float(cl.iloc[-1] / cl.iloc[-self.momentum_window] - 1.0)
        except Exception:
            return None
        if not np.isfinite(r):
            return None
        return r

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_now = float(spy_close.iloc[-1])
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute KBE, KRE, XLF momentum
            kbe_r = self._mom_return(ctx, _KBE)
            kre_r = self._mom_return(ctx, _KRE)
            xlf_r = self._mom_return(ctx, _XLF)

            if kbe_r is None or kre_r is None or xlf_r is None:
                # Fall back to XLF if any data missing
                if _XLF in live:
                    target[_XLF] = self.exposure
            else:
                regional_composite = 0.5 * (kbe_r + kre_r)
                spread = regional_composite - xlf_r

                if spread > self.spread_threshold:
                    # Regional banks leading — credit expansion phase
                    half = self.exposure / 2.0
                    if _KBE in live:
                        target[_KBE] = half
                    if _KRE in live:
                        target[_KRE] = half
                elif spread < -self.spread_threshold:
                    # Broad financials leading — late-cycle defensive
                    if _XLF in live:
                        target[_XLF] = self.exposure
                else:
                    # Neutral — default to XLF
                    if _XLF in live:
                        target[_XLF] = self.exposure

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


def _universe() -> list[str]:
    return [_SPY, _TLT, _KBE, _KRE, _XLF]


NAME = "opus2_kbe_kre_xlf_subindustry"
HYPOTHESIS = (
    "Sub-industry sub-sector spread within financials: KBE+KRE regional banks composite "
    "vs XLF broad financials 63d momentum spread. Regional-banks-led (+2% spread) hold "
    "KBE 48%+KRE 49% (credit-expansion phase); XLF-led (-2% spread) hold XLF 97% "
    "(late-cycle defensive financials); neutral hold XLF 97%. SPY 200d bear gate to TLT. "
    "Biweekly rebalance. Novel: first sub-industry-within-sector spread on leaderboard."
)

UNIVERSE = _universe

STRATEGY = KbeKreXlfSubIndustry()
