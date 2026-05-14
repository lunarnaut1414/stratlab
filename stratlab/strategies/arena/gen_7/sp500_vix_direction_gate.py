"""SP500 momentum with dual gate: SPY 200d SMA + VIX 20d MA direction.

Hypothesis: hold top-20 SP500 stocks by 63d momentum when SPY is above 200d
SMA AND VIX is below its own 20d MA (VIX in downtrend = falling fear). This
dual gate uses VIX direction (trend) rather than VIX level, which is different
from the existing VIX-level gate (VIX < 25 threshold). TLT defensive when
either condition fails. Inverse-vol weighted. Rebalance every 10 bars.

Rationale: VIX trending below its 20d MA means volatility is falling — the
market is "calming down." This is a more dynamic signal than a static VIX level
threshold because it adapts to the current vol regime. A VIX of 22 trending
down is different from a VIX of 18 trending up. Combined with the SPY 200d SMA
trend gate, this creates a dual-confirmation system that only deploys capital
when both price trend AND volatility trend favor equities.

Distinction from existing strategies:
  - VIX 20d MA direction (not level) as gate — truly novel signal
  - Dual SPY-trend + VIX-direction confirmation (not single VIX threshold)
  - Inverse-vol weighting on SP500 cross-section
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10   # bars
MOMENTUM_WINDOW = 63   # ~3 months
VOL_WINDOW = 20        # inverse-vol weighting
TREND_WINDOW = 200     # SPY 200d SMA
VIX_MA_WINDOW = 20     # VIX direction gate
TOP_K = 20
EXPOSURE = 0.97
_VIX = "^VIX"


class SP500VixDirectionGate(Strategy):
    """SP500 63d momentum with SPY 200d SMA + VIX 20d MA dual gate.
    TLT defensive when either gate fails. Inverse-vol weighted.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        vix_ma_window: int = VIX_MA_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            trend_window=trend_window,
            vix_ma_window=vix_ma_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.vix_ma_window = int(vix_ma_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + self.vix_ma_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # VIX 20d MA direction gate (VIX below its 20d MA = falling vol)
        vix_calm = False
        try:
            vix_hist = ctx.history(_VIX)
            vix_close = vix_hist["close"].dropna()
            if len(vix_close) >= self.vix_ma_window + 1:
                vix_current = float(vix_close.iloc[-1])
                vix_ma = float(vix_close.iloc[-self.vix_ma_window:].mean())
                vix_calm = vix_current < vix_ma  # VIX in downtrend
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        both_gates = spy_bull and vix_calm

        if not both_gates:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K SP500 by 63d momentum, inverse-vol weighted
            need = self.momentum_window + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + self.vol_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 5:
                    continue

                # 63d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", _VIX]


NAME = "sp500_vix_direction_gate"
HYPOTHESIS = (
    "SP500 top-20 by 63d momentum with DUAL gate (SPY 200d SMA AND VIX 20d MA): "
    "hold when SPY > 200d SMA AND VIX < VIX 20d MA (VIX trending down = falling fear regime); "
    "TLT defensive otherwise; inverse-vol weighted; rebalance every 10 bars; "
    "VIX 20d MA direction distinct from VIX level gate"
)

UNIVERSE = _universe

STRATEGY = SP500VixDirectionGate()
