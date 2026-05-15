"""opus-3 gen_10 ensemble #2 — REGIME-GATED ETF allocator (no stock-picking).

Companion to opus3_ensemble_regime_gated_triple. Both ensembles share the
mutually-exclusive regime-gating architecture but answer different questions:

  - Ensemble #1: VIX gate, stock-picking component in calm regime.
                 Tests whether stock-picking edge survives when gated.
  - Ensemble #2 (this): VIX gate, NO stock-picking. Pure ETF allocator
                 across three regimes. Tests whether the regime-gating
                 ALONE (without any stock-selection edge) can match the
                 curated benchmark gen5_ensemble_bond_credit_seasonal
                 (IS 0.68, OOS 0.53, 78% retention).

  If ensemble #2 alone achieves IS Calmar > 0.6 with corr_to_top5 < 0.85,
  it proves the regime gate is the load-bearing mechanism — independent
  of any stock-selection alpha — which is the strongest possible OOS
  resilience claim because OOS regime shifts should affect stock-picking
  more than they affect ETF allocations.

Regime gates (mutually exclusive — exactly one component per bar):
  - Calm bull   : SPY > SPY_200d AND VIX < 17  (~61% IS days)
                   -> A = SPY 60% + IEF 37% (risk-on tilted blend)
  - Other bull  : SPY > SPY_200d AND VIX >= 17 (~30% IS days)
                   -> B = SPY 30% + IEF 67% (defensive bull blend)
  - Bear         : SPY <= SPY_200d              (~9% IS days)
                   -> C = TLT 60% + IEF 37% (full defensive)

No SP500 cross-section; no stock quality filters; no per-stock vol-targeting.
Just three ETF allocations selected by gate state. Maximally orthogonal to
the top-5 SP500-momentum-cluster leaderboard rows.

Rationale per gen_10 phase2 brief:
  "Build a regime-gated ensemble where different components are ACTIVE in
   different regimes so failure modes don't compound."

The gen_8 and gen_9 ensembles each had 2-3 stock-picking-style components
combined via weighted sum — all needed the calm-VIX IS regime to work,
and all broke OOS. This ensemble removes stock-picking entirely so there
is no calm-VIX dependency in any single component.

Tradeoff: this ensemble likely has lower IS Calmar than ensemble #1
(which contains a stock-picking edge) but should have much higher OOS
retention because it doesn't rely on the IS calm-VIX regime to produce
its returns. Both ensembles are submitted so the comparison is direct.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ---------------------------------------------------------------------------
# Regime gate parameters
# ---------------------------------------------------------------------------
SPY_TREND_WINDOW = 200       # SPY 200d SMA: bull vs bear
VIX_THRESHOLD = 17.0         # calm (<17) vs other-bull (>=17)

# ---------------------------------------------------------------------------
# Component A — calm bull risk-on blend
# ---------------------------------------------------------------------------
A_SPY_WEIGHT = 0.60
A_IEF_WEIGHT = 0.37

# ---------------------------------------------------------------------------
# Component B — other-bull defensive blend
# ---------------------------------------------------------------------------
B_SPY_WEIGHT = 0.30
B_IEF_WEIGHT = 0.67

# ---------------------------------------------------------------------------
# Component C — bear blend
# ---------------------------------------------------------------------------
C_TLT_WEIGHT = 0.60
C_IEF_WEIGHT = 0.37

EXPOSURE_CAP = 0.97
REBALANCE_EVERY = 10


class Opus3EnsembleEtfRegimeAllocator(Strategy):
    """Mutually-exclusive VIX-and-SPY200d gated ETF allocator with no stock-picking.

    Three ETF allocations selected by regime state:
      - Calm bull   -> SPY 60% / IEF 37%
      - Other bull  -> SPY 30% / IEF 67%
      - Bear         -> TLT 60% / IEF 37%
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spy_trend_window: int = SPY_TREND_WINDOW,
        vix_threshold: float = VIX_THRESHOLD,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spy_trend_window=spy_trend_window,
            vix_threshold=vix_threshold,
        )
        self.rebalance_every = int(rebalance_every)
        self.spy_trend_window = int(spy_trend_window)
        self.vix_threshold = float(vix_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + 20
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        # VIX gate
        try:
            vix_hist = ctx.history("^VIX")
            vix_close = vix_hist["close"].dropna()
            current_vix = float(vix_close.iloc[-1]) if len(vix_close) > 0 else float("nan")
        except KeyError:
            current_vix = float("nan")

        if not spy_bull:
            target = {"TLT": C_TLT_WEIGHT, "IEF": C_IEF_WEIGHT}
        elif np.isnan(current_vix) or current_vix < self.vix_threshold:
            target = {"SPY": A_SPY_WEIGHT, "IEF": A_IEF_WEIGHT}
        else:
            target = {"SPY": B_SPY_WEIGHT, "IEF": B_IEF_WEIGHT}

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target = {sym: w for sym, w in target.items() if sym in live and w > 0}
        total = sum(target.values())
        if total <= 0:
            return []
        if total > EXPOSURE_CAP:
            scale = EXPOSURE_CAP / total
            target = {k: v * scale for k, v in target.items()}

        target_shares: dict[str, int] = {}
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            n = int(equity * weight / price)
            if n > 0:
                target_shares[sym] = n

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target_shares and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta < -1:
                orders.append(Order(side=OrderSide.SELL, size=abs(delta), symbol=sym))
        for sym, tgt in target_shares.items():
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if delta > 1:
                orders.append(Order(side=OrderSide.BUY, size=delta, symbol=sym))
        return orders


UNIVERSE = ["SPY", "TLT", "IEF", "^VIX"]

NAME = "opus3_ensemble_etf_regime_allocator"
HYPOTHESIS = (
    "Pure ETF regime-allocator (NO stock-picking, NO quality filters): "
    "exactly ONE of {A=SPY 60pct+IEF 37pct (calm bull), B=SPY 30pct+IEF 67pct "
    "(other bull), C=TLT 60pct+IEF 37pct (bear)} active per bar via gate (SPY>200d "
    "AND VIX<17 -> A; SPY>200d AND VIX>=17 -> B; SPY<=200d -> C). Tests whether "
    "regime-gating ALONE — without stock-selection alpha — can match curated "
    "ensemble_bond_credit_seasonal OOS retention; maximally orthogonal to "
    "top-5 stock-picking cluster."
)

STRATEGY = Opus3EnsembleEtfRegimeAllocator()
