"""Risk-Parity ETF Allocator — gen_5 sonnet-3

Hypothesis: Hold SPY/TLT/IAU weighted inversely by 20d realized volatility
(vol-parity / inverse-vol allocation), rebalance monthly.  Volatility-parity
captures cross-asset diversification benefit by giving more weight to
lower-volatility assets, reducing drawdowns relative to equal-weight or
cap-weight allocations.

IS window: 2010-01-01 to 2018-12-31.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "TLT", "IAU"]

_VOL_WINDOW = 20      # days for realized vol estimate
_REBALANCE = 21       # monthly rebalance cadence
_EXPOSURE = 0.97      # fraction of portfolio to invest


class RiskParitySpyTltGld(Strategy):
    """Inverse-volatility allocation across SPY, TLT, and IAU."""

    def __init__(
        self,
        vol_window: int = _VOL_WINDOW,
        rebalance: int = _REBALANCE,
        exposure: float = _EXPOSURE,
    ) -> None:
        super().__init__(vol_window=vol_window, rebalance=rebalance, exposure=exposure)
        self.vol_window = vol_window
        self.rebalance = rebalance
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Need enough bars for vol estimation
        if ctx.idx < self.vol_window + 5:
            return []

        # Only rebalance monthly
        if ctx.idx % self.rebalance != 0:
            return []

        closes = ctx.closes()
        live_closes_dict = {s: float(p) for s, p in closes.items()}
        equity = ctx.portfolio_value(live_closes_dict)
        if equity <= 0:
            return []

        # Compute inverse-volatility weights
        inv_vols: dict[str, float] = {}
        for sym in UNIVERSE:
            hist = ctx.history(sym)
            if len(hist) < self.vol_window + 2:
                continue
            prices = hist["close"].iloc[-self.vol_window - 1 :]
            rets = prices.pct_change().dropna()
            if len(rets) < 5:
                continue
            vol = float(rets.std())
            if vol > 0:
                inv_vols[sym] = 1.0 / vol

        if len(inv_vols) < 2:
            return []

        total_inv_vol = sum(inv_vols.values())
        weights: dict[str, float] = {
            sym: iv / total_inv_vol for sym, iv in inv_vols.items()
        }

        # Compute target share counts
        target: dict[str, int] = {}
        for sym, w in weights.items():
            price = live_closes_dict.get(sym, 0.0)
            if price <= 0:
                continue
            alloc = equity * self.exposure * w
            shares = int(alloc // price)
            if shares > 0:
                target[sym] = shares

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, tgt in target.items():
            current = ctx.position(sym).size
            delta = tgt - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "risk_parity_spy_tlt_gld"
HYPOTHESIS = (
    "Risk-parity ETF allocator: hold SPY/TLT/IAU weighted inversely by 20d realized "
    "volatility, rebalance monthly; volatility-parity captures cross-asset diversification "
    "benefit."
)

STRATEGY = RiskParitySpyTltGld(
    vol_window=20,
    rebalance=21,
    exposure=0.97,
)
