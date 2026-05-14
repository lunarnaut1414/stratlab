"""Inverse-ETF Tail-Hedge Sleeve — gen_8 opus-2 (gap_finder)

Hypothesis: Add a small SH (1x inverse SPY) overlay to a SPY core position
sized by regime, instead of rotating between SPY and TLT/cash. The SH sleeve
acts as a negative-beta tail hedge that pays off in fast crashes (Aug 2011,
Aug 2015, Dec 2018), while TLT/cash defensive sleeves only help in slow
yield-driven sell-offs.

Three regimes:
  - Benign  (SPY > 200d SMA AND VIX < 22)    : SPY 97%
  - Mild    (SPY > 200d SMA AND VIX 22-28)   : SPY 85% + SH 12%
  - Severe  (SPY < 200d SMA OR VIX > 28)     : SPY 50% + TLT 35% + SH 12%

Why this fills a gap (after 4 rounds of arena history):
- No prior strategy uses an INVERSE ETF as a defensive sleeve. All defensives
  on the board are TLT, IEF, SHY, GLD, or 100% cash. SH provides direct
  -1 beta exposure that decorrelates from rate/credit-driven defensives.
- The mid-tier "mild stress" allocation (85% SPY + 12% SH = 73% net long)
  is structurally distinct from the binary on/off and from continuous
  vol-target reductions: it keeps full SPY shares ALSO present, so equity
  upside continues to participate while the SH hedge dampens drawdowns.
- Cache: SH starts 2006-06; IS window 2010-2018 has full coverage.

Weekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
SPY_TREND_WINDOW = 200
VIX_CALM = 22.0
VIX_SEVERE = 28.0
W_BENIGN_SPY = 0.97
W_MILD_SPY = 0.85
W_MILD_SH = 0.12
W_SEVERE_SPY = 0.50
W_SEVERE_TLT = 0.35
W_SEVERE_SH = 0.12
EXPOSURE_CAP = 0.97

_SPY = "SPY"
_TLT = "TLT"
_SH = "SH"
_VIX = "^VIX"


class InverseTailHedgeSleeve(Strategy):
    """Three-regime SPY+SH+TLT sleeve with inverse-ETF tail hedge."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spy_trend_window: int = SPY_TREND_WINDOW,
        vix_calm: float = VIX_CALM,
        vix_severe: float = VIX_SEVERE,
        w_benign_spy: float = W_BENIGN_SPY,
        w_mild_spy: float = W_MILD_SPY,
        w_mild_sh: float = W_MILD_SH,
        w_severe_spy: float = W_SEVERE_SPY,
        w_severe_tlt: float = W_SEVERE_TLT,
        w_severe_sh: float = W_SEVERE_SH,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spy_trend_window=spy_trend_window,
            vix_calm=vix_calm,
            vix_severe=vix_severe,
            w_benign_spy=w_benign_spy,
            w_mild_spy=w_mild_spy,
            w_mild_sh=w_mild_sh,
            w_severe_spy=w_severe_spy,
            w_severe_tlt=w_severe_tlt,
            w_severe_sh=w_severe_sh,
        )
        self.rebalance_every = int(rebalance_every)
        self.spy_trend_window = int(spy_trend_window)
        self.vix_calm = float(vix_calm)
        self.vix_severe = float(vix_severe)
        self.w_benign_spy = float(w_benign_spy)
        self.w_mild_spy = float(w_mild_spy)
        self.w_mild_sh = float(w_mild_sh)
        self.w_severe_spy = float(w_severe_spy)
        self.w_severe_tlt = float(w_severe_tlt)
        self.w_severe_sh = float(w_severe_sh)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + 10
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

        # VIX regime
        vix_level = 18.0  # benign default if unavailable
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= 5:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= 5:
                    # 5-day mean to denoise daily spikes
                    vix_level = float(vix_close.iloc[-5:].mean())
        except Exception:
            pass

        # Determine regime
        target: dict[str, float] = {}
        if not spy_bull or vix_level > self.vix_severe:
            # Severe regime
            if _SPY in live:
                target[_SPY] = self.w_severe_spy
            if _TLT in live:
                target[_TLT] = self.w_severe_tlt
            if _SH in live:
                target[_SH] = self.w_severe_sh
        elif vix_level > self.vix_calm:
            # Mild stress
            if _SPY in live:
                target[_SPY] = self.w_mild_spy
            if _SH in live:
                target[_SH] = self.w_mild_sh
        else:
            # Benign
            if _SPY in live:
                target[_SPY] = self.w_benign_spy

        # Normalize so total <= EXPOSURE_CAP if it overshoots
        total = sum(target.values())
        if total > EXPOSURE_CAP and total > 0:
            scale = EXPOSURE_CAP / total
            target = {s: w * scale for s, w in target.items()}

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
    return [_SPY, _TLT, _SH, _VIX]


NAME = "opus2_inverse_tail_hedge_sleeve"
HYPOTHESIS = (
    "Inverse-ETF tail-hedge sleeve: 3-regime SPY core + SH overlay + TLT defensive. "
    "Benign (SPY>200d AND VIX<22) hold SPY 97%; mild stress (SPY>200d AND VIX 22-28) "
    "hold SPY 85%+SH 12%; severe (SPY<200d OR VIX>28) hold SPY 50%+TLT 35%+SH 12%. "
    "Weekly rebalance. SH (1x inverse SPY) provides direct negative-beta hedge — no "
    "prior leaderboard strategy uses an inverse ETF as defensive sleeve."
)

UNIVERSE = _universe

STRATEGY = InverseTailHedgeSleeve()
