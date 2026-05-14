"""Gold-VIX (^GVZ) Regime Allocator — gen_8 opus-2 (gap_finder)

Hypothesis: Use the CBOE Gold ETF Volatility Index (^GVZ) as a regime signal
distinct from equity-VIX, credit (JNK), rates (TNX), and dollar (UUP). GVZ
measures implied vol of GLD options — it reflects stress in the inflation-
hedge / real-asset complex, which has a different cycle profile than equity
fear or credit stress.

3-tier rotation across QQQ/GLD/IEF based on GVZ 252d rolling percentile:
  - GVZ < 33rd pct (calm gold-vol)  : QQQ 97%
       Stable inflation expectations → growth equities work
  - GVZ 33-67th pct (neutral)       : QQQ 60% + GLD 37%
       Mixed environment → balanced growth + hedge
  - GVZ > 67th pct (elevated)       : IEF 60% + GLD 37%
       Real-asset stress → defensive bonds + safe-haven gold

Why this fills a gap (after 4 rounds):
- NO prior strategy uses ^GVZ. Searched gen_5/6/7/8 strategy modules: zero hits.
- GVZ is distinct from:
    * ^VIX (equity vol — done extensively)
    * ^VVIX (vol-of-vol — done in gen_5 and gen_6)
    * ^OVX, ^EVZ (untouched but commodity/FX-specific)
    * JNK/LQD/HYG (credit)
    * TNX/IRX/TYX/FVX (rates)
    * UUP (dollar)
- Cache: GVZ from 2008-06 — full coverage of 2010-2018 IS window.
- Avoids SP500 stock selection entirely → should yield low corr to top-5
  (similar mechanism to dividend_credit_rotation's corr=0.48).
- Uses ROLLING PERCENTILE (252d) not absolute thresholds, which adapts to
  regime shifts in gold-vol baseline.

Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
PCT_WINDOW = 252
LOW_PCT = 0.33
HIGH_PCT = 0.67
W_CALM_QQQ = 0.97
W_NEUT_QQQ = 0.60
W_NEUT_GLD = 0.37
W_STRESS_IEF = 0.60
W_STRESS_GLD = 0.37
EXPOSURE = 0.97

_QQQ = "QQQ"
_GLD = "GLD"
_IEF = "IEF"
_GVZ = "^GVZ"


class GvzGoldVolRegimeQqq(Strategy):
    """3-tier QQQ/GLD/IEF rotation gated by ^GVZ 252d percentile."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        pct_window: int = PCT_WINDOW,
        low_pct: float = LOW_PCT,
        high_pct: float = HIGH_PCT,
        w_calm_qqq: float = W_CALM_QQQ,
        w_neut_qqq: float = W_NEUT_QQQ,
        w_neut_gld: float = W_NEUT_GLD,
        w_stress_ief: float = W_STRESS_IEF,
        w_stress_gld: float = W_STRESS_GLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            pct_window=pct_window,
            low_pct=low_pct,
            high_pct=high_pct,
            w_calm_qqq=w_calm_qqq,
            w_neut_qqq=w_neut_qqq,
            w_neut_gld=w_neut_gld,
            w_stress_ief=w_stress_ief,
            w_stress_gld=w_stress_gld,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.pct_window = int(pct_window)
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.w_calm_qqq = float(w_calm_qqq)
        self.w_neut_qqq = float(w_neut_qqq)
        self.w_neut_gld = float(w_neut_gld)
        self.w_stress_ief = float(w_stress_ief)
        self.w_stress_gld = float(w_stress_gld)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.pct_window + 10
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

        # GVZ percentile signal
        regime: str = "neutral"
        try:
            gvz_hist = ctx.history(_GVZ)
            if gvz_hist is not None and len(gvz_hist) >= self.pct_window + 2:
                gvz_close = gvz_hist["close"].dropna()
                if len(gvz_close) >= self.pct_window:
                    window = gvz_close.iloc[-self.pct_window:].values
                    current = float(gvz_close.iloc[-1])
                    rank_pct = float(np.mean(window <= current))
                    if rank_pct < self.low_pct:
                        regime = "calm"
                    elif rank_pct > self.high_pct:
                        regime = "stress"
                    else:
                        regime = "neutral"
        except Exception:
            pass

        target: dict[str, float] = {}
        if regime == "calm":
            if _QQQ in live:
                target[_QQQ] = self.w_calm_qqq
        elif regime == "stress":
            if _IEF in live:
                target[_IEF] = self.w_stress_ief
            if _GLD in live:
                target[_GLD] = self.w_stress_gld
        else:
            if _QQQ in live:
                target[_QQQ] = self.w_neut_qqq
            if _GLD in live:
                target[_GLD] = self.w_neut_gld

        # Cap total
        total = sum(target.values())
        if total > self.exposure and total > 0:
            scale = self.exposure / total
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
    return [_QQQ, _GLD, _IEF, _GVZ]


NAME = "opus2_gvz_gold_vol_regime_qqq"
HYPOTHESIS = (
    "Gold-VIX (^GVZ) 252d-percentile regime allocator across QQQ/GLD/IEF: "
    "GVZ<33rd pct (calm gold-vol) hold QQQ 97%; GVZ 33-67th pct (neutral) hold "
    "QQQ 60%+GLD 37%; GVZ>67th pct (elevated) hold IEF 60%+GLD 37%. Biweekly "
    "rebalance. Novel: no prior arena strategy uses ^GVZ; rolling-percentile "
    "thresholds adapt to regime baseline; pure ETF allocator with no SP500 "
    "stock selection should yield low corr to top-5."
)

UNIVERSE = _universe

STRATEGY = GvzGoldVolRegimeQqq()
