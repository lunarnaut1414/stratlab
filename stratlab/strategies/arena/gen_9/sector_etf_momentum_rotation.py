"""gen_9 sonnet-6 — Sector ETF Momentum Rotation

Hypothesis: Each month, rank 7 broad sector ETFs (XLK, XLF, XLE, XLI, XLU,
XLY, XLB) by 63d absolute momentum. Hold top-2 sectors with positive momentum,
inverse-vol weighted. If fewer than 2 sectors have positive momentum, hold SPY.
SPY 200d SMA bear gate → TLT.

Rationale: Sector ETF rotation is structurally different from individual stock
selection — it selects entire economic sectors rather than individual companies.
The return stream of holding 2 sector ETFs is less correlated with the SP500
momentum cluster because:
1. Sector ETFs are more concentrated (30-90 stocks vs 15 individual picks)
2. The winning sector changes over economic cycles (tech → financials → energy)
3. Inverse-vol weighting gives more weight to lower-vol defensive sectors

Key distinction from existing leaderboard strategies:
- Not SPY/QQQ/TLT/GLD ETF rotation (too few options)
- Not SP500 individual stock selection (too correlated)
- Sector ETFs used as the HOLDING not the SIGNAL

Sectors used: XLK, XLF, XLE, XLI, XLU, XLY, XLB
  (7 sectors with full IS coverage — XLRE/XLV/XLP excluded for IS gaps)

Monthly rebalance (21 bars) generates sufficient trades over IS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21   # monthly
MOMENTUM_WINDOW = 63
VOL_WINDOW = 21
TOP_K = 2
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_TREND_WINDOW = 200

# Sector ETFs with full IS coverage (pre-2010 inception)
_SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLU", "XLY", "XLB"]


class SectorEtfMomentumRotation(Strategy):
    """Sector ETF top-2 momentum rotation with inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        trend_window: int = _TREND_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
            trend_window=trend_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.trend_window = int(trend_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute momentum and vol for each sector ETF
            scores: dict[str, float] = {}
            vols: dict[str, float] = {}

            for sec in _SECTORS:
                try:
                    h = ctx.history(sec)
                except KeyError:
                    continue
                if h is None or len(h) < self.momentum_window + 2:
                    continue
                c = h["close"].dropna()
                if len(c) < self.momentum_window + 1:
                    continue

                mom = float(c.iloc[-1] / c.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(mom):
                    continue
                scores[sec] = mom

                if len(c) >= self.vol_window + 1:
                    log_r = np.log(c.values[1:] / c.values[:-1])
                    rv = float(np.std(log_r[-self.vol_window:]) * np.sqrt(252))
                    if rv > 0 and np.isfinite(rv):
                        vols[sec] = rv

            # Filter to positive momentum only
            positive = {s: v for s, v in scores.items() if v > 0}

            if len(positive) < 2:
                # Fewer than 2 positive sectors: hold SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                k = min(self.top_k, len(positive))
                ranked = sorted(positive, key=positive.__getitem__, reverse=True)[:k]

                # Inverse-vol weighting
                inv_vols = {}
                for sec in ranked:
                    vol = vols.get(sec, 0.15)
                    inv_vols[sec] = 1.0 / max(vol, 0.01)

                total_inv = sum(inv_vols.values())
                if total_inv > 0:
                    for sec in ranked:
                        w = inv_vols[sec] / total_inv * self.exposure
                        if sec in live:
                            target[sec] = w
                else:
                    per_weight = self.exposure / len(ranked)
                    for sec in ranked:
                        if sec in live:
                            target[sec] = per_weight

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


# Universe includes sector ETFs as holdable assets + signal-only SPY/TLT
UNIVERSE = _SECTORS + [_SPY, _TLT]

NAME = "sector_etf_momentum_rotation"
HYPOTHESIS = (
    "Sector ETF momentum rotation among XL* sectors: each month hold top-2 sector ETFs "
    "(XLK/XLF/XLE/XLI/XLU/XLY/XLB) by 63d absolute return with positive momentum; "
    "inverse-vol weighted; if fewer than 2 positive sectors hold SPY; SPY 200d bear gate "
    "to TLT; monthly rebalance; selects sector ETFs not individual stocks"
)

STRATEGY = SectorEtfMomentumRotation()
