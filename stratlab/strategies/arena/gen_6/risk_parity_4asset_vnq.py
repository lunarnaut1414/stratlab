"""4-Asset Risk Parity with REITs — gen_6 sonnet-4

Hypothesis: Extend the gen_5 3-asset risk parity (SPY/TLT/GLD) to 4 assets
by adding VNQ (REIT index) for additional diversification. Real estate has
distinct return drivers (cap rates, rental income, credit) from equities,
rates, and gold. Inverse-volatility weighting. Monthly rebalance.

VIX-adaptive cash buffer: reserve 10% cash when VIX > 25 (reduce exposure
slightly during fear spikes). This is mild — not a full rotation like gen_5
strategies, just a buffer.

Distinct from gen5_risk_parity_spy_tlt_gld:
  - 4 assets vs 3 (adds VNQ)
  - Uses IAU instead of GLD (same gold exposure, different ETF)
  - VIX-adaptive cash buffer (gen_5 has no VIX gate)
  - Biweekly (10-bar) rebalance vs monthly (21-bar)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ASSETS = ["SPY", "TLT", "GLD", "VNQ"]
VOL_WINDOW = 21
REBALANCE = 10       # biweekly
VIX_THRESHOLD = 25.0
BASE_EXPOSURE = 0.97
STRESSED_EXPOSURE = 0.87  # 10% cash buffer when VIX > 25
_VIX = "^VIX"


class RiskParity4AssetVNQ(Strategy):
    """4-asset risk parity (SPY/TLT/GLD/VNQ), biweekly rebalance, mild VIX buffer."""

    def __init__(
        self,
        vol_window: int = VOL_WINDOW,
        rebalance: int = REBALANCE,
        vix_threshold: float = VIX_THRESHOLD,
        base_exposure: float = BASE_EXPOSURE,
        stressed_exposure: float = STRESSED_EXPOSURE,
    ) -> None:
        super().__init__(
            vol_window=vol_window,
            rebalance=rebalance,
            vix_threshold=vix_threshold,
            base_exposure=base_exposure,
            stressed_exposure=stressed_exposure,
        )
        self.vol_window = int(vol_window)
        self.rebalance = int(rebalance)
        self.vix_threshold = float(vix_threshold)
        self.base_exposure = float(base_exposure)
        self.stressed_exposure = float(stressed_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.vol_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # VIX level for exposure scaling
        vix_level = 20.0
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        exposure = self.stressed_exposure if vix_level >= self.vix_threshold else self.base_exposure

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute inverse-vol weights
        need = self.vol_window + 2
        prices = ctx.closes_window(need)
        if len(prices) < self.vol_window:
            return []

        inv_vols: dict[str, float] = {}
        for sym in ASSETS:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.vol_window + 1:
                continue
            # Use last vol_window prices
            tail = col.values[-self.vol_window - 1:]
            logr = np.log(tail[1:] / tail[:-1])
            rv = float(np.std(logr, ddof=1)) * np.sqrt(252)  # annualized
            if rv <= 1e-6 or not np.isfinite(rv):
                continue
            inv_vols[sym] = 1.0 / rv

        if len(inv_vols) < 2:
            return []

        total_inv_vol = sum(inv_vols.values())
        target: dict[str, float] = {}
        for sym, iv in inv_vols.items():
            weight = iv / total_inv_vol * exposure
            target[sym] = weight

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "risk_parity_4asset_vnq"
HYPOTHESIS = (
    "4-asset risk parity SPY/TLT/GLD/VNQ: weight inversely to 21d realized vol with biweekly rebalance; "
    "include REIT (VNQ) as 4th asset class for diversification; "
    "VIX-adaptive cash buffer (87% exposure when VIX>25, 97% otherwise)"
)

UNIVERSE = ["SPY", "TLT", "GLD", "VNQ", _VIX]

STRATEGY = RiskParity4AssetVNQ()
