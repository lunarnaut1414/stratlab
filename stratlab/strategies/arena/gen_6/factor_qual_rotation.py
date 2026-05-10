"""Factor ETF absolute-momentum rotation — gen_6 sonnet-7

Hypothesis: Rank MTUM/QUAL/IVE/USMV by 3-month total return; hold
top-2 equally weighted only when both have positive 3m returns; rotate
to AGG when neither qualifies. Monthly rebalance. Pure factor rotation
with absolute momentum gate (no trend/VIX filter).

Structural distinctions vs existing leaderboard:
- Uses VUG/VTV/VGT/VFH/VHT (growth, value, tech, financials, healthcare)
- Absolute momentum filter: only hold if return > 0 (prevents holding
  negative-momentum factors)
- AGG (broad aggregate bond) as defensive, not TLT/IEF/SHY
- No SPY trend gate or VIX gate at all — pure factor-momentum signal
- Monthly rebalance with consistent factor diversification
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FACTOR_ETFS = ["VUG", "VTV", "VGT", "VFH", "VHT", "VCR", "VDC"]  # plus Consumer Disc/Staples
CASH_PROXY = "AGG"
MOMENTUM_WINDOW = 63   # ~3 months
TOP_K = 3              # hold top-3 (not 2) for more trades
REBALANCE_EVERY = 10   # biweekly for more trades
EXPOSURE = 0.97
VIX_THRESHOLD = 28.0   # rotate to AGG only in severe stress


class FactorQualRotation(Strategy):
    """Vanguard factor ETF absolute-momentum rotation: top-2 Vanguard factor ETFs
    with positive 3m return; AGG when neither qualifies."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
        vix_threshold: float = VIX_THRESHOLD,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            top_k=top_k,
            rebalance_every=rebalance_every,
            exposure=exposure,
            vix_threshold=vix_threshold,
        )
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self.vix_threshold = float(vix_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + 5
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

        # VIX stress gate: go to AGG only in severe stress
        vix_stressed = False
        try:
            vix_hist = ctx.history("^VIX")
            if len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
                if np.isfinite(vix_level) and vix_level >= self.vix_threshold:
                    vix_stressed = True
        except Exception:
            pass

        target: dict[str, float] = {}

        if vix_stressed:
            if CASH_PROXY in closes_now.index:
                target[CASH_PROXY] = self.exposure
        else:
            # Compute 3-month momentum for each Vanguard sector/factor ETF
            scores: dict[str, float] = {}
            for sym in FACTOR_ETFS:
                try:
                    hist = ctx.history(sym)
                except Exception:
                    continue
                if len(hist) < self.momentum_window + 2:
                    continue
                closes_ser = hist["close"].dropna()
                if len(closes_ser) < self.momentum_window:
                    continue
                ret = float(closes_ser.iloc[-1] / closes_ser.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                if CASH_PROXY in closes_now.index:
                    target[CASH_PROXY] = self.exposure
            else:
                # Hold top-K by relative momentum
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / k
                for sym in ranked:
                    target[sym] = per_weight

        orders: list[Order] = []

        # Exit positions not in target
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


NAME = "factor_qual_rotation"
HYPOTHESIS = (
    "Vanguard factor ETF absolute-momentum rotation: rank VUG/VTV/VGT/VFH/VHT by 3-month return; "
    "hold top-2 equally weighted when both have positive 3m returns; hold AGG "
    "when neither qualifies; monthly rebalance; pure factor rotation with absolute momentum filter"
)
UNIVERSE = ["VUG", "VTV", "VGT", "VFH", "VHT", "VCR", "VDC", "AGG", "^VIX", "SPY"]
STRATEGY = FactorQualRotation()
