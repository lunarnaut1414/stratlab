"""gen_9 sonnet-6 — Cross-Asset ETF Momentum with Volatility Parity Sizing

Hypothesis: Rank a curated set of broad-market and bond ETFs by 63d absolute
momentum. Hold top-3 ETFs with positive momentum, weighted by inverse realized
volatility (vol-parity allocation). If all have negative momentum, hold TLT.
Weekly rebalance.

Universe: SPY, QQQ, IWM, TLT, LQD, AGG
  - 3 equity ETFs (large/mega/small cap)
  - 3 fixed income ETFs (long-bond, investment-grade corp, agg bond)
  - All have full IS coverage (pre-2010 inception)

Rationale: Most leaderboard strategies select individual SP500 stocks (which
all correlate with each other). This strategy stays entirely in ETFs — a
fundamentally different return stream. The vol-parity sizing ensures no single
asset dominates: TLT (low vol) gets more weight than IWM (high vol) when both
are in top-3. Absolute momentum filter (only positive-return ETFs) provides
basic trend protection without a VIX/MA regime gate.

The ETF-only approach means:
  - Much lower turnover than stock selection
  - Different loss-mode correlation from SP500 momentum strategies
  - Natural diversification across equity styles and duration

Weekly rebalance (every 5 bars) to generate sufficient trades over IS window.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOMENTUM_WINDOW = 63      # ~3 months for ranking
VOL_WINDOW = 21           # 21d for inverse-vol sizing
TOP_K = 3
EXPOSURE = 0.97

# Curated ETF universe — all have pre-2010 inception
_ETFS = ["SPY", "QQQ", "IWM", "TLT", "LQD", "AGG"]
_DEFENSIVE = "TLT"


class EtfVolparityMomentum(Strategy):
    """Cross-asset ETF momentum rotation with inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.vol_window) + 10
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

        # Compute momentum and vol for each ETF in universe
        need = max(self.momentum_window, self.vol_window) + 5
        scores: dict[str, float] = {}
        vols: dict[str, float] = {}

        for sym in _ETFS:
            try:
                h = ctx.history(sym)
            except KeyError:
                continue
            if h is None or len(h) < self.momentum_window + 2:
                continue
            c = h["close"].dropna()
            if len(c) < self.momentum_window + 1:
                continue

            # 63d momentum
            mom = float(c.iloc[-1] / c.iloc[-self.momentum_window] - 1.0)
            if not np.isfinite(mom):
                continue
            scores[sym] = mom

            # 21d realized vol for inverse-vol sizing
            if len(c) >= self.vol_window + 1:
                log_r = np.log(c.values[1:] / c.values[:-1])
                rv = float(np.std(log_r[-self.vol_window:]) * np.sqrt(252))
                if rv > 0 and np.isfinite(rv):
                    vols[sym] = rv
                else:
                    vols[sym] = 0.15  # fallback annualized vol

        target: dict[str, float] = {}

        if not scores:
            # No data: hold cash
            pass
        else:
            # Filter to positive momentum only
            positive = {s: v for s, v in scores.items() if v > 0}

            if not positive:
                # All negative momentum: defensive TLT
                if _DEFENSIVE in live:
                    target[_DEFENSIVE] = self.exposure
            else:
                # Hold top-K by momentum (from positive set)
                k = min(self.top_k, len(positive))
                ranked = sorted(positive, key=positive.__getitem__, reverse=True)[:k]

                # Inverse-vol weighting
                inv_vols = {}
                for sym in ranked:
                    vol = vols.get(sym, 0.15)
                    inv_vols[sym] = 1.0 / max(vol, 0.01)

                total_inv = sum(inv_vols.values())
                if total_inv > 0:
                    for sym in ranked:
                        w = inv_vols[sym] / total_inv * self.exposure
                        if sym in live:
                            target[sym] = w
                else:
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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


# ETF-only strategy — no individual stocks needed
UNIVERSE = _ETFS

NAME = "etf_volparity_momentum"
HYPOTHESIS = (
    "Cross-asset ETF momentum with vol-parity sizing across SPY/QQQ/IWM/TLT/LQD/AGG: "
    "each week rank by 63d momentum, hold top-3 ETFs with positive absolute momentum, "
    "inverse-vol weighted (vol-parity allocation); if all negative momentum hold TLT only; "
    "weekly rebalance; all-ETF portfolio, never holds SP500 individual stocks"
)

STRATEGY = EtfVolparityMomentum()
