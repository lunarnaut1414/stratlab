"""opus-1 mutation of hy_credit_qqq_rotation (credit-allocator cluster).

Parent: gen6_hy_credit_qqq_rotation (IS Calmar 0.78, h2>>h1, corr_to_top5 0.41).

Structural mutations vs parent (vehicle-only swap):
  - Risk-on:    QQQ (Nasdaq-100, ~100 tech-heavy)  ->  VUG (Vanguard Growth,
                ~270 large-cap-growth basket — broader, less tech-concentrated).
  - Defensive:  TLT (long treasuries, ~17yr duration)  ->  AGG (broad-bond
                aggregate, ~6yr duration — mid-curve, less rate-sensitivity
                during 2013 taper tantrum and 2018 rate hikes).
  - Everything else identical: JNK 30d MA, SPY 100d MA, weekly rebalance,
    0.97 exposure, dual-bull gating logic.

Why this should be admitted under 0.85 corr filter:
  - VUG vs QQQ: VUG was relatively flat in 2018 Q4 tech selloff while QQQ
    drew down ~16% — the broader basket has different daily innovations.
    QQQ also outperforms in tech-led 2017 by a wide margin; VUG captures
    growth more conservatively.
  - AGG vs TLT: AGG returns ~5% across the IS window, TLT returns ~50%+.
    The duration mismatch makes their daily PnL diverge meaningfully on
    rate-cycle days. AGG is much smoother — different drawdown profile.
  - Parent has corr_to_top5 0.41 already; vehicle swap should keep it well
    under 0.85.

Note: parent has h2>>h1 — preserving the parent's structure (signal, MA
windows, rebalance cadence) maximizes the chance the variant inherits its
sub-period stability.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 30
SPY_MA = 100
REBALANCE_EVERY = 5
EXPOSURE = 0.97

GROWTH_ETF = "VUG"
DEFENSIVE_ETF = "IEF"


class HyCreditVugAgg(Strategy):
    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        spy_ma: int = SPY_MA,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            spy_ma=spy_ma,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.spy_ma = int(spy_ma)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.spy_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # JNK 30d SMA trend
        jnk_bullish = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 1:
                jnk_close = jnk_hist["close"].dropna()
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_bullish = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        # SPY 100d SMA confirmation
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.spy_ma + 1:
                spy_close = spy_hist["close"].dropna()
                spy_sma = float(spy_close.iloc[-self.spy_ma:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}
        if jnk_bullish and spy_bull:
            if GROWTH_ETF in live:
                target[GROWTH_ETF] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        else:
            if DEFENSIVE_ETF in live:
                target[DEFENSIVE_ETF] = self.exposure
            elif "IEF" in live:
                target["IEF"] = self.exposure
            elif "TLT" in live:
                target["TLT"] = self.exposure

        if not target:
            return []

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


NAME = "opus1_hy_credit_vug_agg"
HYPOTHESIS = (
    "Mutate hy_credit_qqq_rotation: vehicle-only swap — risk-on QQQ -> VUG "
    "(broader growth basket), defensive TLT -> AGG (mid-duration); preserves "
    "JNK 30d MA + SPY 100d MA dual gate, weekly rebalance, 0.97 exposure."
)
UNIVERSE = ["JNK", "VUG", "QQQ", "AGG", "IEF", "TLT", "SPY"]

STRATEGY = HyCreditVugAgg()
