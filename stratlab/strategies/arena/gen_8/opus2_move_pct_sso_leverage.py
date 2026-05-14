"""MOVE Bond-Vol Percentile Gating SSO Leverage — gen_8 opus-2 (gap_finder)

Hypothesis: The ^MOVE index (option-implied volatility of US Treasury futures)
measures bond-market risk that is structurally distinct from VIX (equity vol).
When MOVE is at extreme lows, the bond market is signalling "no rate-shock
risk on the horizon" — which historically aligns with periods of low
cross-asset risk and supports leveraged equity exposure.

3-tier rotation gated by MOVE 90d rolling percentile:

  - MOVE < 25th pct (calm bond mkt)   : SSO 50% + SPY 47% (mild 2x SPY tilt)
       No rate-shock risk → can carry SSO leverage decay safely
  - MOVE 25-75th pct (normal)         : SPY 97%
       Track market — no edge from MOVE level
  - MOVE > 75th pct (bond stress)     : SPY 50% + IEF 47%
       Defensive INTERMEDIATE bonds (NOT TLT — when MOVE is high, long
       duration is fragile to rate moves; IEF has lower duration risk)

Why this fills a gap:
- gen_5 opus1_etf_move_factor_rotation used MOVE to gate MTUM/QUAL factor
  ETFs (not SPY/leverage). Distinct construction.
- gen_5 leveraged_etf_momentum used SSO with a TREND signal (SPY 200d MA
  + VIX 22d MA + 21d/63d momentum), not bond-vol regime.
- The combination "bond-vol regime → leverage tilt" is structurally novel.
- Use of IEF (not TLT) in stress tier is also distinct — most strategies
  reflexively go to TLT, but TLT is precisely what gets hurt when MOVE
  spikes. IEF preserves defensive intent without rate-vol exposure.
- Cache: ^MOVE from 2002-11; SSO from 2006-06 — full IS coverage.

Weekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
PCT_WINDOW = 90
LOW_PCT = 0.25
HIGH_PCT = 0.75
W_CALM_SSO = 0.50
W_CALM_SPY = 0.47
W_NORMAL_SPY = 0.97
W_STRESS_SPY = 0.50
W_STRESS_IEF = 0.47
EXPOSURE = 0.97

_SPY = "SPY"
_SSO = "SSO"
_IEF = "IEF"
_MOVE = "^MOVE"


class MovePctSsoLeverage(Strategy):
    """3-tier SPY/SSO/IEF allocator gated by MOVE 90d percentile."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        pct_window: int = PCT_WINDOW,
        low_pct: float = LOW_PCT,
        high_pct: float = HIGH_PCT,
        w_calm_sso: float = W_CALM_SSO,
        w_calm_spy: float = W_CALM_SPY,
        w_normal_spy: float = W_NORMAL_SPY,
        w_stress_spy: float = W_STRESS_SPY,
        w_stress_ief: float = W_STRESS_IEF,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            pct_window=pct_window,
            low_pct=low_pct,
            high_pct=high_pct,
            w_calm_sso=w_calm_sso,
            w_calm_spy=w_calm_spy,
            w_normal_spy=w_normal_spy,
            w_stress_spy=w_stress_spy,
            w_stress_ief=w_stress_ief,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.pct_window = int(pct_window)
        self.low_pct = float(low_pct)
        self.high_pct = float(high_pct)
        self.w_calm_sso = float(w_calm_sso)
        self.w_calm_spy = float(w_calm_spy)
        self.w_normal_spy = float(w_normal_spy)
        self.w_stress_spy = float(w_stress_spy)
        self.w_stress_ief = float(w_stress_ief)
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

        # MOVE percentile signal
        regime = "normal"
        try:
            mv_hist = ctx.history(_MOVE)
            if mv_hist is not None and len(mv_hist) >= self.pct_window + 2:
                mv_close = mv_hist["close"].dropna()
                if len(mv_close) >= self.pct_window:
                    window = mv_close.iloc[-self.pct_window:].values
                    current = float(mv_close.iloc[-1])
                    rank_pct = float(np.mean(window <= current))
                    if rank_pct < self.low_pct:
                        regime = "calm"
                    elif rank_pct > self.high_pct:
                        regime = "stress"
                    else:
                        regime = "normal"
        except Exception:
            pass

        target: dict[str, float] = {}
        if regime == "calm":
            if _SSO in live:
                target[_SSO] = self.w_calm_sso
            if _SPY in live:
                target[_SPY] = self.w_calm_spy
        elif regime == "stress":
            if _SPY in live:
                target[_SPY] = self.w_stress_spy
            if _IEF in live:
                target[_IEF] = self.w_stress_ief
        else:
            if _SPY in live:
                target[_SPY] = self.w_normal_spy

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
    return [_SPY, _SSO, _IEF, _MOVE]


NAME = "opus2_move_pct_sso_leverage"
HYPOTHESIS = (
    "MOVE bond-vol 90d-percentile gating SSO leverage sleeve: MOVE<25th pct (calm "
    "bond market) hold SSO 50%+SPY 47% (mild 2x leverage tilt); 25-75th pct (normal) "
    "hold SPY 97%; >75th pct (bond stress) hold SPY 50%+IEF 47% (intermediate bonds, "
    "not TLT — IEF less duration-sensitive when MOVE spikes). Weekly rebalance. "
    "Novel: bond-vol-as-leverage-gate; gen_5 MOVE strategy used factor ETFs not "
    "leverage; IEF defensive choice avoids rate-vol exposure."
)

UNIVERSE = _universe

STRATEGY = MovePctSsoLeverage()
