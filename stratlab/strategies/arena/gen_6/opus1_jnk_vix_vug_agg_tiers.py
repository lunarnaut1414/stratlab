"""opus-1 mutation of jnk_vix_dual_gate_qqq (credit-allocator cluster).

Parent: gen6_jnk_vix_dual_gate_qqq (IS Calmar 0.86, h2>h1, corr_to_top5 0.71).

Structural mutations vs parent (vehicle & threshold swap):
  - Risk-on equity:  QQQ  ->  VUG (Vanguard growth ETF, broader basket).
  - Defensive bucket: SHY 50% + TLT 47%  ->  SHY 50% + AGG 47% (mid-duration
                      not long-duration; different rate-sensitivity profile).
  - VIX tier 1 cutoff: 20  ->  18 (slightly tighter — only fires on the very
                      calmest regime).
  - VIX tier 3 cutoff: 28  ->  25 (faster trip into defensive — quicker risk-off).
  - Rebalance:        weekly (5)  ->  biweekly (10).

Why this should be admitted under 0.85 corr filter:
  - VUG vs QQQ: VUG is broad large-cap growth (~270 names), QQQ is Nasdaq-100
    (~100 tech-heavy names). Daily innovations diverge meaningfully; VUG was
    less hit by 2018 Q4 tech selloff and the 2014-15 oil-driven semis dip.
  - AGG vs TLT: AGG is mid-duration broad bond (~6yr duration), TLT is long
    treasury (~17yr duration). Different sensitivity to 2013 taper tantrum
    and 2018 rate cycle.
  - The combination of two vehicle swaps + two threshold tweaks shifts both
    the *trip dates* and the *daily return path* enough that 0.85 corr is
    plausible while preserving the parent's stable h2>h1 signal.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 20
VIX_CALM_THRESHOLD = 18.0
VIX_CAUTION_THRESHOLD = 25.0
REBALANCE_EVERY = 10
EXPOSURE = 0.97


class JnkVixVugAggTiers(Strategy):
    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        vix_calm: float = VIX_CALM_THRESHOLD,
        vix_caution: float = VIX_CAUTION_THRESHOLD,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            vix_calm=vix_calm,
            vix_caution=vix_caution,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.vix_calm = float(vix_calm)
        self.vix_caution = float(vix_caution)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.jnk_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        credit_bullish = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma + 1:
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    credit_bullish = jnk_now > jnk_ma_val
        except KeyError:
            pass

        vix_level = 20.0
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= 2:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= 1:
                    vix_level = float(vix_close.iloc[-1])
        except KeyError:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not credit_bullish:
            # Risk-off
            if "SHY" in live:
                target["SHY"] = 0.50 * self.exposure
            if "AGG" in live:
                target["AGG"] = 0.47 * self.exposure
            elif "IEF" in live:
                target["IEF"] = 0.47 * self.exposure
            if not target and "SHY" in live:
                target["SHY"] = self.exposure
        elif vix_level < self.vix_calm:
            # Tier 1: credit good + low VIX → VUG (growth)
            if "VUG" in live:
                target["VUG"] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        elif vix_level < self.vix_caution:
            # Tier 2: credit good + moderate VIX → SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
            elif "VUG" in live:
                target["VUG"] = self.exposure
        else:
            # Tier 3: credit good but VIX elevated → 60/40 SPY/AGG
            if "SPY" in live:
                target["SPY"] = 0.60 * self.exposure
            if "AGG" in live:
                target["AGG"] = 0.40 * self.exposure
            elif "IEF" in live:
                target["IEF"] = 0.40 * self.exposure

        if not target:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


NAME = "opus1_jnk_vix_vug_agg_tiers"
HYPOTHESIS = (
    "Mutate jnk_vix_dual_gate_qqq: VUG growth replaces QQQ in tier 1; AGG mid-"
    "duration replaces TLT in defensive bucket; VIX tier thresholds 18/25 "
    "(parent 20/28); biweekly rebalance; vehicle+threshold swap shifts trip "
    "dates and daily PnL path."
)
UNIVERSE = ["JNK", "VUG", "QQQ", "SPY", "SHY", "AGG", "IEF", "^VIX"]

STRATEGY = JnkVixVugAggTiers()
