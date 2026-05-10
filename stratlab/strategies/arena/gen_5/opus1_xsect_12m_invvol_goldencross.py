"""opus-1 mutation of vix_gated_sp500_momentum (xsect_sp500_mom cluster).

Structural mutations vs parent (gen5_vix_gated_sp500_momentum, IS Calmar 0.82):
  - Momentum window: 63d  ->  126d skip 21d (6-1 month classic).
  - Top-K:           15    ->  20.
  - Sizing:          equal ->  inverse 20d realized vol (vol-targeted weights).
  - Rebalance:       10    ->  21 (monthly, less churn).
  - Gate:            VIX<25  -> SPY 200d SMA only (no VIX, no fast SMA).
  - Defensive:       SHY+TLT -> IEF only (single mid-duration treasury).

Strategy designed to differ structurally from the saturated VIX-gated /
golden-cross / 21d-skip-1 cluster (corr_to_top5 routinely > 0.65). Inverse-vol
weighting in particular changes the daily return path materially because
high-vol momentum names (e.g. NVDA-style) get a smaller position than the
parent's equal-weight rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
MOM_LOOKBACK = 126         # 6 months
MOM_SKIP = 21              # skip last 1 month
VOL_WINDOW = 20            # for inverse-vol weights
TOP_K = 20
TREND_WINDOW = 200
EXPOSURE = 0.97


class XSect12mInvVolGoldenCross(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = self.mom_lookback + self.mom_skip + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue
                end_idx = -self.mom_skip
                start_idx = -(self.mom_lookback + self.mom_skip)
                p_end = float(col.iloc[end_idx])
                p_start = float(col.iloc[start_idx])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < self.top_k:
                return []

            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[: self.top_k]
            iv_sum = sum(inv_vols[s] for s in ranked)
            if iv_sum <= 0:
                return []
            for sym in ranked:
                target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "opus1_xsect_12m_invvol_goldencross"
HYPOTHESIS = (
    "Mutate vix_gated_sp500_momentum: top-20 SP500 by 6-1 month momentum "
    "(126d skip 21d), inverse-vol weighted, monthly rebalance, gated on SPY "
    "200d SMA; defensive bucket IEF only (single mid-duration treasury)."
)

UNIVERSE = _universe

STRATEGY = XSect12mInvVolGoldenCross()
