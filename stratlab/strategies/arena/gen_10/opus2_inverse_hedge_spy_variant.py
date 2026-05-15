"""SH (inverse SPY) tail-hedge sleeve, SPY-core variant.

Hypothesis (opus-2, gen_10 gap_finder):
    SPY-core version of the gen9_opus2_inverse_etf_tail_hedge mechanism with
    tighter stress trigger and larger hedge sleeve:

    - Calm regime (VIX <= 25 AND SPY > 100d SMA):
        95% SPY (no hedge).
    - Stress regime (VIX > 25 OR SPY < 100d SMA):
        75% SPY + 20% SH (inverse SPY) — net ~55% beta.
    Rebalance every 5 bars.

Why this is an OPEN frontier (phase2_brief explicit ask):
    gen9_opus2_inverse_etf_tail_hedge scored IS Calmar 0.62 with loss-mode-corr
    0.52 — distinctive low-corr profile. Phase2 brief explicitly suggested:
    "try a DIFFERENT trigger threshold (VIX>25 instead of >22, or SPY 100d SMA
    instead of 200d), or different sleeve weight (20% instead of 15%)."

    This variant differs from gen9_opus2 along THREE axes simultaneously:
      gen9_opus2 had: QQQ core, VIX>22, QQQ 200d SMA, 80/15 hedge.
      this strategy: SPY core, VIX>25, SPY 100d SMA, 75/20 hedge.

    Mechanism remains: explicit short-side hedge OVERLAY on always-on long
    equity exposure, structurally different from defensive *routing into
    bonds*. Loss-mode corr should remain low because the hedge fires on
    days that hurt SPY-momentum strategies the most.

  SH carries ~1% annual borrow + tracking drag, so the strategy must earn
  enough calm-state SPY upside to offset hedge cost. SPY core (vs QQQ)
  has lower IS upside but more stable drawdowns — this trades calm-state
  ceiling for stress-state floor.

Distinct from:
  - gen9_opus2_inverse_etf_tail_hedge: QQQ core, VIX>22, QQQ 200d, 80/15.
  - All gen_10 strategies: NO inverse-ETF hedge sleeve at all.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "SH", "^VIX"]

VIX_THRESHOLD = 25.0
SMA_PERIOD = 100
REBALANCE_EVERY = 5
W_SPY_CALM = 0.95
W_SPY_STRESS = 0.75
W_SH_STRESS = 0.20


class InverseHedgeSpyVariant(Strategy):
    def __init__(
        self,
        vix_threshold: float = VIX_THRESHOLD,
        sma_period: int = SMA_PERIOD,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            vix_threshold=vix_threshold,
            sma_period=sma_period,
            rebalance_every=rebalance_every,
        )
        self.vix_threshold = float(vix_threshold)
        self.sma_period = int(sma_period)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sma_period + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        vix_level = float("nan")
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if spy_hist is None or len(spy_hist) < self.sma_period + 1:
            return []
        sc = spy_hist["close"].dropna()
        spy_now = float(sc.iloc[-1])
        spy_sma = float(sc.iloc[-self.sma_period:].mean())
        spy_below = np.isfinite(spy_now) and np.isfinite(spy_sma) and spy_now < spy_sma
        vix_high = np.isfinite(vix_level) and vix_level > self.vix_threshold

        if vix_high or spy_below:
            target = {"SPY": W_SPY_STRESS, "SH": W_SH_STRESS}
        else:
            target = {"SPY": W_SPY_CALM}

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
            tgt = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


NAME = "opus2_inverse_hedge_spy_variant"
HYPOTHESIS = (
    "SH inverse-ETF tail-hedge sleeve variant: in calm regime (VIX<=25 AND SPY>100d SMA) hold 95pct SPY; "
    "in stress (VIX>25 OR SPY<100d SMA) hold 75pct SPY+20pct SH (inverse SPY) — net ~55pct beta; "
    "rebalance every 5 bars — DIFFERS from gen9_opus2 (VIX>22, QQQ core, QQQ 200d SMA, 80/15 sleeve); "
    "this is SPY core + tighter VIX trigger + larger hedge sleeve to test whether SPY+stricter+bigger "
    "beats gen9 IS 0.62"
)

STRATEGY = InverseHedgeSpyVariant()
