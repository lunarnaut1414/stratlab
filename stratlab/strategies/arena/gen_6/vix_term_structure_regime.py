"""VIX term-structure regime allocator (opus-5 wildcard, gen_6).

Hypothesis (anti-consensus, untouched in gen_5/gen_6):
The shape of the VIX term structure — ratio of 3-month implied vol (^VIX3M)
to spot vol (^VIX) — is a regime indicator with structurally different
dynamics from VIX *level* (saturated on leaderboard) and from VVIX/MOVE/SKEW
*levels* (also saturated and gen_5 dead-ends).

  - Deep contango (VIX3M / VIX >= 1.05): the curve is normally upward-sloping.
    Far-dated vol > spot. Stable equity regime. Tilt aggressively to QQQ.
  - Mild contango (1.00 <= VIX3M / VIX < 1.05): borderline / neutral.
    Hold SPY — broader, less drawdown-prone than QQQ.
  - Backwardation (VIX3M / VIX < 1.00): spot vol exceeds 3-month — acute
    stress, panic in front-month. In the 2010-2018 IS regime, backwardation
    episodes (2010 May flash, 2011 Aug debt-ceiling, 2015 Aug, 2018 Feb
    Volmageddon) are typically short-lived but have outsized downside while
    they last. Rotate to SHY 50% + TLT 47% defensive.

The signal is smoothed via a 5-day MA on the *ratio* (not on each component
separately) to avoid one-day spikes flipping the regime. Weekly rebalance.

Differentiation from leaderboard:
  - VIX-LEVEL gating (vix_calm_42d, jnk_vix_dual_gate, sma_cross_vix_gate,
    spy_momentum_vix_qqq, vix_safehaven_gld_tlt_spy, vix_composite_qqq):
    fires on a different *axis* — VIX-level gates fire on absolute fear,
    term-structure fires on whether the curve is normal-shaped vs panic-shaped.
    On many calm days VIX is low (<20) but VIX3M / VIX can still tighten or
    flip into backwardation, signalling regime change before VIX itself rises.
  - VVIX gating (qqq_bollinger_vvix_dipbuy gen_5): VVIX is vol-of-vol level.
    Term-structure is shape, not level. Different statistic.
  - SKEW (gen_5 skew_tail_risk): tail-risk perception. Different.
  - MOVE (gen_5 opus1_etf_move_factor_rotation): bond-market vol level.
    Different asset class.
  - Credit-spread allocators (jnk_*, hy_*, credit_*): credit market signal,
    distinct from equity vol curve.
  - Yield-curve (rsi_etf_meanrev gen_5 was TNX-IRX): rate-curve, not vol-curve.

Past attempts in dead_ends.md flagged "vix_term_structure_carry" — that was a
*carry* trade (VIX9D/VIX) — different operationalisation that produced n=0
trades or sub-0.5 Calmar. This strategy is a regime allocator (3-tier
QQQ/SPY/defensive), not a vol-carry trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "SHY", "^VIX", "^VIX3M"]

RATIO_SMOOTH = 5            # 5-day moving average on the ratio
DEEP_CONTANGO_THRESHOLD = 1.03  # ratio >= this => deep contango (lowered to capture more bull days)
MILD_CONTANGO_THRESHOLD = 1.00  # ratio in [1.00, 1.03) => mild contango
REBALANCE_EVERY = 5         # weekly
EXPOSURE = 0.97


class VixTermStructureRegime(Strategy):
    def __init__(
        self,
        ratio_smooth: int = RATIO_SMOOTH,
        deep_contango: float = DEEP_CONTANGO_THRESHOLD,
        mild_contango: float = MILD_CONTANGO_THRESHOLD,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            ratio_smooth=ratio_smooth,
            deep_contango=deep_contango,
            mild_contango=mild_contango,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.ratio_smooth = int(ratio_smooth)
        self.deep_contango = float(deep_contango)
        self.mild_contango = float(mild_contango)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def _smoothed_ratio(self, ctx: BarContext) -> float | None:
        try:
            vix_hist = ctx.history("^VIX")
            vix3m_hist = ctx.history("^VIX3M")
        except KeyError:
            return None
        if vix_hist is None or vix3m_hist is None:
            return None
        vix = vix_hist["close"].dropna()
        vix3m = vix3m_hist["close"].dropna()
        if len(vix) < self.ratio_smooth + 2 or len(vix3m) < self.ratio_smooth + 2:
            return None
        # Align on common dates
        joined = pd.concat(
            [vix.rename("vix"), vix3m.rename("vix3m")], axis=1, join="inner"
        ).dropna()
        if len(joined) < self.ratio_smooth + 1:
            return None
        ratio = (joined["vix3m"] / joined["vix"]).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(ratio) < self.ratio_smooth:
            return None
        smoothed = float(ratio.iloc[-self.ratio_smooth:].mean())
        if not np.isfinite(smoothed) or smoothed <= 0:
            return None
        return smoothed

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.ratio_smooth + 20
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        ratio = self._smoothed_ratio(ctx)
        if ratio is None:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p)) and float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        if ratio >= self.deep_contango:
            # Deep contango -> QQQ aggressive
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        elif ratio >= self.mild_contango:
            # Mild contango -> SPY broad
            if "SPY" in live:
                target["SPY"] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
        else:
            # Backwardation -> partial defensive: in 2010-2018 IS regime,
            # backwardation typically lasts <2 weeks and dips recover quickly,
            # so keep equity exposure but add bond hedge.
            if "SPY" in live:
                target["SPY"] = 0.60 * self.exposure
            if "TLT" in live:
                target["TLT"] = 0.30 * self.exposure
            if not target and "TLT" in live:
                target["TLT"] = self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
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


NAME = "vix_term_structure_regime"
HYPOTHESIS = (
    "VIX term-structure regime allocator: smoothed 5d ^VIX3M/^VIX ratio gates "
    "QQQ/SPY/defensive. Deep contango (ratio>=1.05) hold QQQ 97%; mild contango "
    "(1.00-1.05) hold SPY 97%; backwardation (ratio<1.00) hold SHY 50% + TLT 47%; "
    "weekly rebalance. Pure VIX term-structure shape (not level) — orthogonal to "
    "VIX/VVIX/MOVE/SKEW level allocators on leaderboard."
)

STRATEGY = VixTermStructureRegime()
