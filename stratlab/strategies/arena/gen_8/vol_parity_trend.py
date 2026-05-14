"""Cross-Asset Vol-Parity Trend Following — gen_8 sonnet-3

Hypothesis: Hold SPY/QQQ/TLT/GLD when each is above its 100d SMA; size each
inversely to its 20d realized volatility (vol-parity risk budgeting); if fewer
than 2 assets are trending, revert to SPY+TLT equal-weight; weekly rebalance.

Rationale: Multi-asset trend following is well-documented but most implementations
use equal-weight or fixed allocations. Vol-parity sizing dynamically allocates
more capital to lower-volatility assets in uptrend. Adding SHY as cash proxy
when no assets trend creates a sensible fallback. This differs from the existing
rp_credit_tilt (which is always invested) and the gen5 risk parity (which uses
SPY/TLT/GLD and is also always invested). This strategy ONLY holds assets in
uptrend, making it trend-following rather than risk-parity.

IS window: 2010-2018 (9 years).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5         # Weekly
TREND_WINDOW = 100          # 100d SMA for trend filter
VOL_WINDOW = 20             # 20d realized vol for sizing
MIN_ASSETS_TRENDING = 2     # Minimum assets in uptrend; else use SPY+TLT
EXPOSURE = 0.97

RISKY_ASSETS = ["SPY", "QQQ", "TLT", "GLD"]
FALLBACK_A = "SPY"
FALLBACK_B = "TLT"
CASH_PROXY = "SHY"


class VolParityTrend(Strategy):
    """Vol-parity sized multi-asset trend following."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        min_assets_trending: int = MIN_ASSETS_TRENDING,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vol_window=vol_window,
            min_assets_trending=min_assets_trending,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.min_assets_trending = int(min_assets_trending)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vol_window + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

        need = self.trend_window + 5
        prices_df = ctx.closes_window(need)
        if len(prices_df) < self.trend_window:
            return []

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Find trending assets and compute inverse-vol weights
        trending: list[str] = []
        inv_vols: dict[str, float] = {}

        for sym in RISKY_ASSETS:
            if sym not in prices_df.columns:
                continue
            col = prices_df[sym].dropna()
            if len(col) < self.trend_window + 1:
                continue

            # Check uptrend
            sma = float(col.iloc[-self.trend_window:].mean())
            price = float(col.iloc[-1])
            if price <= sma:
                continue

            # Compute 20d realized vol
            if len(col) < self.vol_window + 1:
                continue
            tail = col.iloc[-(self.vol_window + 1):]
            log_rets = np.log(tail.values[1:] / tail.values[:-1])
            rv = float(np.std(log_rets))
            if rv < 1e-8 or not np.isfinite(rv):
                continue

            trending.append(sym)
            inv_vols[sym] = 1.0 / rv

        target: dict[str, float] = {}

        if len(trending) >= self.min_assets_trending:
            # Vol-parity weights across trending assets
            total_inv_vol = sum(inv_vols[s] for s in trending)
            if total_inv_vol > 0:
                for sym in trending:
                    target[sym] = self.exposure * inv_vols[sym] / total_inv_vol
        else:
            # Fewer than min_assets trending — use SPY+TLT equal fallback
            fallback = []
            for sym in [FALLBACK_A, FALLBACK_B]:
                if sym in live:
                    fallback.append(sym)
            if fallback:
                per_slot = self.exposure / len(fallback)
                for sym in fallback:
                    target[sym] = per_slot
            else:
                # Last resort: cash
                if CASH_PROXY in live:
                    target[CASH_PROXY] = self.exposure

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


NAME = "vol_parity_trend"
HYPOTHESIS = (
    "Cross-asset trend-following with vol-parity sizing across 5 assets: hold "
    "SPY/QQQ/TLT/GLD each when price above its 100d SMA; position-size each inversely "
    "to its 20d realized volatility (vol-parity); if fewer than 2 assets are trending "
    "revert to SPY+TLT equal-weight; weekly rebalance; 5-asset diversified momentum "
    "with risk-budgeting across uncorrelated asset classes"
)

UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "SHY"]

STRATEGY = VolParityTrend()
