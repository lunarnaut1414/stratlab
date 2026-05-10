"""Credit-spread gated sector ETF rotation.

Hypothesis: Use JNK/LQD z-score as a macro risk barometer.
- z > +0.5 (credit expanding, risk-on):  top-2 cyclical sectors by 21d
  momentum from {XLK, XLY, XLI, XLF} equally weighted + XLV (30% hedge).
- -0.5 <= z <= +0.5 (neutral):  top-2 from all 9 sectors by 21d momentum.
- z < -0.5 (credit contracting, risk-off): XLV 40% + XLP 35% + TLT 22%.

Rebalance every 5 bars (weekly).
SPY 200d SMA secondary gate: if SPY below SMA, override to risk-off bucket.

Structural distinctions vs existing leaderboard:
- Uses JNK/LQD credit spread (z-score window 90d) as primary regime signal
- Routes exposure through sector ETFs (not individual stocks)
- Combines credit macro signal with sector momentum signal
- Defensive allocation via defensive sectors (XLV/XLP) + TLT, not full bond rotation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

CYCLICAL_SECTORS = ["XLK", "XLY", "XLI", "XLF"]
ALL_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEFENSIVE_ALLOC = {"XLV": 0.40, "XLP": 0.35, "TLT": 0.22}

Z_HIGH = 0.5
Z_LOW = -0.5
Z_WINDOW = 90
MOMENTUM_WINDOW = 21
REBALANCE_EVERY = 5
TREND_WINDOW = 200
EXPOSURE = 0.97


class CreditGatedSectorRotation(Strategy):
    """JNK/LQD credit-spread z-score gated sector ETF rotation."""

    def __init__(
        self,
        z_window: int = Z_WINDOW,
        z_high: float = Z_HIGH,
        z_low: float = Z_LOW,
        momentum_window: int = MOMENTUM_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            z_window=z_window,
            z_high=z_high,
            z_low=z_low,
            momentum_window=momentum_window,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.z_window = int(z_window)
        self.z_high = float(z_high)
        self.z_low = float(z_low)
        self.momentum_window = int(momentum_window)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def _credit_zscore(self, ctx: BarContext) -> float | None:
        """Compute current JNK/LQD ratio z-score (90d window)."""
        try:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
        except Exception:
            return None
        if len(jnk_hist) < self.z_window + 5 or len(lqd_hist) < self.z_window + 5:
            return None
        jnk_c = jnk_hist["close"].dropna()
        lqd_c = lqd_hist["close"].dropna()
        # align on common index
        ratio = (jnk_c / lqd_c).dropna()
        if len(ratio) < self.z_window:
            return None
        window = ratio.iloc[-self.z_window:]
        mean = float(window.mean())
        std = float(window.std())
        if std <= 0:
            return None
        return float((window.iloc[-1] - mean) / std)

    def _sector_momentum(self, ctx: BarContext, sector_list: list[str]) -> dict[str, float]:
        """Return momentum scores for available sector ETFs."""
        scores: dict[str, float] = {}
        for sym in sector_list:
            try:
                hist = ctx.history(sym)
                c = hist["close"].dropna()
                if len(c) >= self.momentum_window:
                    ret = float(c.iloc[-1] / c.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret
            except Exception:
                pass
        return scores

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.z_window, self.trend_window, self.momentum_window) + 10
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

        # SPY trend gate
        bull = True
        try:
            spy_hist = ctx.history("SPY")
            spy_c = spy_hist["close"].dropna()
            if len(spy_c) >= self.trend_window:
                bull = float(spy_c.iloc[-1]) > float(spy_c.iloc[-self.trend_window:].mean())
        except Exception:
            pass

        # Credit regime
        z = self._credit_zscore(ctx)

        target: dict[str, float] = {}

        if not bull or (z is not None and z < self.z_low):
            # Risk-off: defensive sectors + bond
            for sym, wt in DEFENSIVE_ALLOC.items():
                if sym in live:
                    target[sym] = wt * self.exposure
        elif z is not None and z > self.z_high:
            # Risk-on: top-3 cyclical sectors + XLV as 4th diversifier
            scores = self._sector_momentum(ctx, CYCLICAL_SECTORS)
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            top3 = ranked[:3]
            if top3:
                per_wt = (self.exposure * 0.75) / len(top3)
                for sym in top3:
                    target[sym] = per_wt
                # Add XLV as defensive diversifier in aggressive state
                if "XLV" in live:
                    target["XLV"] = self.exposure * 0.22
        else:
            # Neutral: top-3 from all sectors by momentum
            scores = self._sector_momentum(ctx, ALL_SECTORS)
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            top3 = ranked[:3]
            if top3:
                per_wt = self.exposure / len(top3)
                for sym in top3:
                    target[sym] = per_wt

        orders: list[Order] = []

        # Sell positions not in target
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


UNIVERSE = [
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "TLT", "SHY", "SPY", "JNK", "LQD"
]
NAME = "credit_gated_sector_rotation"
HYPOTHESIS = (
    "Credit-spread gated sector ETF rotation: JNK/LQD 90d z-score regime; "
    "z>+0.5 hold top-3 cyclical sectors + XLV hedge; neutral hold top-3 all sectors; "
    "z<-0.5 or SPY below 200d hold defensive (XLV+XLP+TLT)"
)
STRATEGY = CreditGatedSectorRotation()
