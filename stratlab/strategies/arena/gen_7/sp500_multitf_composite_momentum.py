"""SP500 multi-timeframe momentum composite with JNK+SPY dual-gate.

Hypothesis: Rank SP500 stocks by a weighted composite of three momentum
windows (21d, 63d, 126d with weights 1:2:3) to smooth out short-term noise
while still capturing medium-term momentum signals. Use JNK trend (credit
conditions) and SPY 200d SMA (equity trend) as dual gatekeepers:

  - BOTH JNK > 30d SMA AND SPY > 200d SMA: hold top-15 SP500 by composite score
  - Either gate fails: rotate to IEF (mid-duration bonds) defensively

Composite score = (1/6)*ret_21d + (2/6)*ret_63d + (3/6)*ret_126d
(weights 1:2:3 give more influence to medium/longer windows, reducing noise)

Rationale: Single-window momentum (e.g., 63d-only) can pick stocks that had
a strong burst but are now mean-reverting. The composite blends short-term
strength, intermediate momentum, and sustained trend into one signal, selecting
stocks with momentum consistent across multiple horizons. The dual JNK+SPY gate
is the same approach as gen6_nearhi_momentum_quality but using a different
ranking signal (composite vs near-high proximity).

Inverse-vol sizing reduces concentration in high-vol winners.
Monthly rebalance (21 bars) limits turnover.

Distinct from existing strategies:
  - Unique composite (21d+63d+126d weighted) scoring not used anywhere
  - JNK credit gate + SPY trend gate (same dual gate as hy_credit_qqq_rotation)
    but applied to SP500 stock selection vs ETF rotation
  - IEF defensive (not TLT or SHY)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
TOP_K = 15
VOL_WINDOW = 20
TREND_WINDOW = 200
JNK_MA = 30
EXPOSURE = 0.97

# Momentum windows and their weights (sum to 6)
MOM_WINDOWS = [(21, 1), (63, 2), (126, 3)]
MAX_WINDOW = 126

_SPY = "SPY"
_JNK = "JNK"


class SP500MultiTFCompositeMomentum(Strategy):
    """Multi-timeframe composite momentum on SP500 with JNK+SPY dual gate.

    Ranks stocks by weighted blend of 21d, 63d, 126d returns. Inverse-vol
    sized. Dual JNK+SPY gate; IEF defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        jnk_ma: int = JNK_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            top_k=top_k,
            vol_window=vol_window,
            trend_window=trend_window,
            jnk_ma=jnk_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.jnk_ma = int(jnk_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MAX_WINDOW + max(self.trend_window, self.jnk_ma) + 10
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

        # SPY 200d SMA gate
        spy_bull = False
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # JNK 30d SMA credit gate
        credit_ok = False
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    credit_ok = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull or not credit_ok:
            # Defensive: IEF
            if "IEF" in live:
                target["IEF"] = self.exposure
        else:
            # Risk-on: compute composite momentum for all SP500 stocks
            need = MAX_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < MAX_WINDOW:
                if "IEF" in live:
                    target["IEF"] = self.exposure
            else:
                total_weight = sum(w for _, w in MOM_WINDOWS)
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < MAX_WINDOW + 1:
                        continue

                    # Compute composite score
                    composite = 0.0
                    valid = True
                    for window, weight in MOM_WINDOWS:
                        if len(col) < window + 1:
                            valid = False
                            break
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-(window + 1)])
                        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                            valid = False
                            break
                        ret = (p_end / p_start) - 1.0
                        if not np.isfinite(ret):
                            valid = False
                            break
                        composite += (weight / total_weight) * ret

                    if not valid or not np.isfinite(composite):
                        continue

                    # Inverse-vol weight
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = composite
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    # Fallback to IEF
                    if "IEF" in live:
                        target["IEF"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        if "IEF" in live:
                            target["IEF"] = self.exposure
                    else:
                        for sym in ranked:
                            target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
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


NAME = "sp500_multitf_composite_momentum"
HYPOTHESIS = (
    "SP500 multi-timeframe momentum composite with JNK+SPY dual-gate: rank SP500 stocks by "
    "weighted composite of 21d+63d+126d returns (weights 1:2:3), hold top-15 when JNK above "
    "30d SMA AND SPY above 200d SMA; inverse-vol weighted; IEF defensive; monthly rebalance; "
    "avoids short-term noise bias of single-window approaches"
)


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY", "JNK"]


UNIVERSE = _universe

STRATEGY = SP500MultiTFCompositeMomentum()
