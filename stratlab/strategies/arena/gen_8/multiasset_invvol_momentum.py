"""Multi-Asset Inverse-Vol Weighted Momentum — gen_8 sonnet-10

Hypothesis: Rank a diverse set of ETFs (SPY, QQQ, TLT, GLD, IEF, IWM, EEM)
by 42-day return. Hold the top-3 ETFs with POSITIVE absolute momentum, weighted
inversely by their 21-day realized volatility (risk-parity sizing within the
momentum winners). Any unfilled slot goes to SHY.

Rebalance every 10 bars (biweekly).

Rationale: This combines two orthogonal signals:
1. Cross-sectional momentum: identifies which asset class is leading
2. Risk-parity sizing: vol-weights so higher-vol winners get smaller allocations

The universe is diverse across equity, bond, commodity, international, and gold —
so the winning assets shift substantially across different regimes (equity-bull,
risk-off, inflationary, etc.). This produces very different return patterns than:
- SP500 stock selection (always in equities when not defensive)
- Single-factor momentum ETF rotation (always picks single winner)
- Pure risk-parity (always holds everything proportionally)

No SPY trend gate — the absolute-momentum filter (return > 0) handles bear markets
naturally by routing to SHY or defensive assets that happen to have positive momentum
(TLT, GLD in certain regimes).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 42      # ~2 months
VOL_WINDOW = 21           # 21d realized vol for sizing
TOP_K = 3
EXPOSURE = 0.97
_SHY = "SHY"

# Diverse multi-asset ETF universe — all with long IS-window coverage
_ASSETS = ["SPY", "QQQ", "TLT", "GLD", "IEF", "IWM", "EEM"]


class MultiAssetInvVolMomentum(Strategy):
    """Top-3 multi-asset ETFs by momentum, inverse-vol weighted."""

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

        # Load price window for momentum and vol
        need = max(self.momentum_window, self.vol_window) + 5
        prices = ctx.closes_window(need)
        if len(prices) < self.momentum_window:
            return []

        # Compute momentum scores (filter: positive momentum only)
        momentum_scores: dict[str, float] = {}
        inv_vols: dict[str, float] = {}

        for sym in _ASSETS:
            if sym not in live:
                continue
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window + 1:
                continue

            # 42d return
            ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
            if not np.isfinite(ret) or ret <= 0:
                continue  # only positive-momentum assets

            momentum_scores[sym] = ret

            # 21d realized vol (annualized)
            recent = col.iloc[-self.vol_window:]
            if len(recent) < self.vol_window:
                continue
            log_rets = np.log(recent.values[1:] / recent.values[:-1])
            ann_vol = float(np.std(log_rets) * np.sqrt(252))
            if ann_vol > 0.001:
                inv_vols[sym] = 1.0 / ann_vol

        if not momentum_scores:
            # Everything has negative momentum — go fully SHY
            target: dict[str, float] = {}
            if _SHY in live:
                target[_SHY] = self.exposure
        else:
            # Pick top-K by momentum score
            k = min(self.top_k, len(momentum_scores))
            ranked = sorted(momentum_scores, key=momentum_scores.__getitem__, reverse=True)[:k]

            # Compute inverse-vol weights for the selected assets
            # Only use inv_vol for assets where vol was computed
            weights_raw: dict[str, float] = {}
            for sym in ranked:
                weights_raw[sym] = inv_vols.get(sym, 1.0)

            total_inv_vol = sum(weights_raw.values())
            if total_inv_vol <= 0:
                # Fall back to equal weight
                per_weight = self.exposure / len(ranked)
                weights_norm = {sym: per_weight for sym in ranked}
            else:
                weights_norm = {
                    sym: (w / total_inv_vol) * self.exposure
                    for sym, w in weights_raw.items()
                }

            # Fill remaining slots with SHY
            n_remaining = self.top_k - len(ranked)
            per_slot = self.exposure / self.top_k
            if n_remaining > 0 and _SHY in live:
                weights_norm[_SHY] = per_slot * n_remaining

            target = weights_norm

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


UNIVERSE = _ASSETS + [_SHY]

NAME = "multiasset_invvol_momentum"
HYPOTHESIS = (
    "Multi-asset inverse-vol weighted momentum: rank SPY/QQQ/TLT/GLD/IEF/IWM/EEM by "
    "42d return, hold top-3 with positive momentum weighted inversely by 21d realized "
    "volatility; rebalance every 10 bars; SHY for unfilled slots; combines momentum "
    "ranking with risk-parity vol-weighting across diverse assets without SPY trend gate"
)

STRATEGY = MultiAssetInvVolMomentum()
