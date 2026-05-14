"""opus-1 / gen_8 — VIX Percentile-Rank Gated QQQ/GLD Blend

Mutation of gen8_vix_gated_qqq_gld_blend (IS Calmar 0.52, h2 0.39,
loss_mode_corr 0.91 — fragile parent).

Parent uses fixed absolute VIX thresholds (16, 22, 30) to tier QQQ/GLD/SPY/TLT
allocations. The brief explicitly flags this as regime-fragile: absolute
thresholds will trigger different states in different regimes (e.g. the
2017-low-vol regime kept VIX < 16 for months, while 2010-12 had VIX > 22
frequently). h2 weakness (0.39) supports this diagnosis.

This variant replaces absolute thresholds with VIX 252d-percentile-rank
thresholds (p25/p50/p75). The signal is self-normalizing: regardless of the
prevailing VIX regime, ~25% of bars trigger each tier (well-defined frequency).

Tiers:
  - VIX pct rank in [0, 25]: calm-relative — QQQ 89% + GLD 11%
  - VIX pct rank in (25, 50]: normal-relative — QQQ 72% + GLD 21% + IEF 7%
  - VIX pct rank in (50, 75]: elevated-relative — SPY 52% + TLT 30% + GLD 18%
  - VIX pct rank in (75, 100]: stressed-relative — TLT 62% + GLD 38%
SPY 200d outer bear gate: force defensive TLT+GLD.

Same blends as parent — only the tier-assignment function changes. Goal:
more stable h1/h2 split and lower loss_mode_corr.

Weekly rebalance, ETF-only universe.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
TREND_WINDOW = 200
VIX_PCT_WINDOW = 252
PCT_LOW = 0.25
PCT_MED = 0.50
PCT_HIGH = 0.75
EXPOSURE = 0.97
_VIX = "^VIX"
_QQQ = "QQQ"
_SPY = "SPY"
_TLT = "TLT"
_GLD = "GLD"
_IEF = "IEF"


class VixPctrankQQQGLDBlend(Strategy):
    """VIX percentile-rank gated QQQ/GLD blend: 4 tiers by 252d rank."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vix_pct_window: int = VIX_PCT_WINDOW,
        pct_low: float = PCT_LOW,
        pct_med: float = PCT_MED,
        pct_high: float = PCT_HIGH,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vix_pct_window=vix_pct_window,
            pct_low=pct_low,
            pct_med=pct_med,
            pct_high=pct_high,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vix_pct_window = int(vix_pct_window)
        self.pct_low = float(pct_low)
        self.pct_med = float(pct_med)
        self.pct_high = float(pct_high)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.vix_pct_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

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

        # VIX percentile rank vs trailing 252d
        vix_pct = 0.5  # default to median if signal unavailable
        try:
            vix_hist = ctx.history(_VIX)
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= self.vix_pct_window:
                window = vix_close.iloc[-self.vix_pct_window:].values
                current = float(vix_close.iloc[-1])
                vix_pct = float(np.mean(window < current))
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        raw_target: dict[str, float] = {}

        if not bull or vix_pct > self.pct_high:
            raw_target[_TLT] = 0.62
            raw_target[_GLD] = 0.38
        elif vix_pct > self.pct_med:
            raw_target[_SPY] = 0.52
            raw_target[_TLT] = 0.30
            raw_target[_GLD] = 0.18
        elif vix_pct > self.pct_low:
            raw_target[_QQQ] = 0.72
            raw_target[_GLD] = 0.21
            raw_target[_IEF] = 0.07
        else:
            raw_target[_QQQ] = 0.89
            raw_target[_GLD] = 0.11

        total_w = sum(raw_target.values())
        target = {
            s: (w / total_w) * self.exposure
            for s, w in raw_target.items()
            if s in live
        }

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


UNIVERSE = [_QQQ, _SPY, _TLT, _GLD, _IEF, _VIX]

NAME = "opus1_vix_pctrank_qqq_gld_blend"
HYPOTHESIS = (
    "Mutation of vix_gated_qqq_gld_blend: replace absolute VIX thresholds (16/22/30) with "
    "VIX 252d percentile-rank tiers (p25/p50/p75); same QQQ/GLD/SPY/TLT/IEF blends per tier; "
    "self-normalizing signal for more stable h1/h2; weekly rebalance; SPY 200d outer bear gate"
)

STRATEGY = VixPctrankQQQGLDBlend()
