"""Multi-asset real-asset momentum rotator.

Hypothesis: rank GLD/TLT/PDBC/USO/SLV by 42d momentum; hold top-2 with positive
absolute momentum, inverse-vol weighted; IEF defensive when all negative; VIX
z-score gate avoids commodity exposure in vol spikes; weekly rebalance.

Rationale: Real assets (gold, commodities, inflation-sensitive bonds) have
low correlation to each other and to equity momentum strategies. By selecting
the best-performing 2 of these 5 real-asset ETFs using absolute momentum, we
capture the "best of breed" in inflation/commodity regimes while avoiding assets
in downtrends. The VIX z-score gate protects against crisis periods when
correlations spike. This is a pure real-asset strategy with no individual
stock selection — orthogonal to all SP500 momentum strategies on the leaderboard.

Key distinctions:
  - Real-asset-only universe (GLD/TLT/PDBC/USO/SLV) not equity
  - Absolute momentum filter (must have positive returns)
  - Inverse-vol weighting on just 2 winners
  - VIX z-score gate (not VIX level) avoids vol-spike regime
  - IEF (not SHY) as intermediate-duration defensive
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # bars (~weekly)
MOMENTUM_WINDOW = 42      # ~2 months
VOL_WINDOW = 20           # for inverse-vol weights
VIX_Z_WINDOW = 60         # z-score baseline for VIX
VIX_Z_THRESHOLD = 1.5    # avoid commodities when VIX is this many std devs above mean
TOP_K = 2
EXPOSURE = 0.97

REAL_ASSETS = ["GLD", "TLT", "DBC", "USO", "SLV"]
DEFENSIVE = "IEF"
_VIX = "^VIX"


class RealAssetMomentumRotator(Strategy):
    """Real-asset momentum: top-2 of GLD/TLT/PDBC/USO/SLV by 42d return,
    absolute momentum filter, inverse-vol sized, VIX z-score gate.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        vix_z_window: int = VIX_Z_WINDOW,
        vix_z_threshold: float = VIX_Z_THRESHOLD,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            vix_z_window=vix_z_window,
            vix_z_threshold=vix_z_threshold,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.vix_z_window = int(vix_z_window)
        self.vix_z_threshold = float(vix_z_threshold)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.vix_z_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # VIX z-score gate — go defensive if VIX is extremely elevated
        stressed = False
        try:
            vix_hist = ctx.history(_VIX)
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= self.vix_z_window + 5:
                vix_window = vix_close.iloc[-self.vix_z_window:]
                vix_mean = float(vix_window.mean())
                vix_std = float(vix_window.std())
                current_vix = float(vix_close.iloc[-1])
                if vix_std > 0:
                    vix_z = (current_vix - vix_mean) / vix_std
                    stressed = vix_z > self.vix_z_threshold
        except Exception:
            pass

        target: dict[str, float] = {}

        if stressed:
            # VIX spike: all to defensive IEF
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        else:
            # Score each real asset on momentum + vol
            need = self.momentum_window + self.vol_window + 5
            prices = ctx.closes_window(need)

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in REAL_ASSETS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.vol_window:
                    continue

                # Absolute momentum: must be positive
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue
                if ret <= 0:
                    continue  # absolute momentum filter: skip negative momentum assets

                # Inverse-vol weight
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if not scores:
                # All negative momentum: go to IEF
                if DEFENSIVE in closes_now.index:
                    target[DEFENSIVE] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    if DEFENSIVE in closes_now.index:
                        target[DEFENSIVE] = self.exposure
                else:
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


UNIVERSE = REAL_ASSETS + [DEFENSIVE, "SPY", _VIX]

NAME = "real_asset_momentum_rotator"
HYPOTHESIS = (
    "GLD/TLT/PDBC multi-commodity real-asset rotator: rank GLD/TLT/PDBC/USO/SLV by 42d momentum; "
    "hold top-2 with positive absolute momentum, inverse-vol weighted; IEF defensive when all negative; "
    "^VIX z-score gate avoids commodity exposure in vol spikes; weekly rebalance"
)

STRATEGY = RealAssetMomentumRotator()
