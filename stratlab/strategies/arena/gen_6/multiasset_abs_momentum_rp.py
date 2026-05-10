"""Multi-asset absolute momentum with risk-parity sizing.

Hypothesis:
  Dual test on each asset in {QQQ, GLD, TLT, SPY}: is 6-month (126d) return
  positive? (absolute momentum test). Among the assets with positive momentum,
  size positions inversely proportional to 20d realized volatility (risk parity).
  When no asset passes the absolute momentum test, hold IEF as cash proxy.
  Rebalance monthly (every 21 bars).

Rationale:
  Faber (2010) and Antonacci (2014) both show that simple absolute momentum
  (return > 0 over a lookback) on multi-asset ETFs produces strong
  risk-adjusted returns by avoiding asset classes in trending decline.
  The cross-asset diversification (equities + gold + bonds) combined with
  inverse-vol weighting creates a different daily return path than pure
  equity momentum strategies.

  Unlike gen5_risk_parity_spy_tlt_gld (which always holds all 3 assets
  regardless of momentum direction), this strategy EXCLUDES assets with
  negative 6m absolute momentum. This means during the 2013-2014 gold
  sell-off, GLD would be excluded, concentrating in the remaining
  positive-momentum assets.

Diversification vs leaderboard:
  - Risk-parity (gen5): always holds SPY+TLT+GLD regardless of direction.
    This strategy is conditional on positive absolute momentum.
  - All SP500 equity strategies: this uses QQQ/GLD/TLT cross-asset
    rotation — fundamentally different asset universe and mechanism.
  - Credit spread strategies: this uses absolute price momentum,
    not spread-based regime signals.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Multi-asset universe for absolute momentum test
ASSETS = ["QQQ", "GLD", "TLT", "SPY"]
DEFENSIVE = "IEF"      # hold when no asset has positive momentum

LOOKBACK = 126         # 6-month absolute momentum window
VOL_WINDOW = 20        # for inverse-vol risk parity sizing
REBALANCE_EVERY = 21   # monthly
EXPOSURE = 0.97


class MultiassetAbsMomentumRP(Strategy):
    """Multi-asset absolute momentum with risk-parity sizing."""

    def __init__(
        self,
        lookback: int = LOOKBACK,
        vol_window: int = VOL_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            lookback=lookback,
            vol_window=vol_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.lookback = int(lookback)
        self.vol_window = int(vol_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.lookback + self.vol_window + 10
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

        need = self.lookback + self.vol_window + 2
        prices = ctx.closes_window(need)
        if len(prices) < self.lookback:
            return []

        # Absolute momentum test for each asset
        eligible: dict[str, float] = {}  # sym -> inv_vol

        for sym in ASSETS:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.lookback + 1:
                continue
            p_start = float(col.iloc[-self.lookback])
            p_end = float(col.iloc[-1])
            if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                continue
            ret_6m = p_end / p_start - 1.0
            # Absolute momentum filter: must have positive 6m return
            if ret_6m <= 0:
                continue
            # Inverse realized volatility for risk-parity sizing
            if len(col) < self.vol_window + 1:
                continue
            tail = col.iloc[-self.vol_window - 1:]
            logr = np.log(tail.values[1:] / tail.values[:-1])
            rv = float(np.std(logr))
            if rv <= 1e-6 or not np.isfinite(rv):
                continue
            eligible[sym] = 1.0 / rv

        target: dict[str, float] = {}

        if len(eligible) == 0:
            # No asset has positive momentum: hold defensive IEF
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        else:
            # Risk-parity: weight inversely proportional to vol
            iv_sum = sum(eligible.values())
            for sym, iv in eligible.items():
                target[sym] = self.exposure * iv / iv_sum

        # Build orders
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


NAME = "multiasset_abs_momentum_rp"
HYPOTHESIS = (
    "Multi-asset absolute momentum with risk-parity sizing: hold QQQ/GLD/TLT/SPY "
    "with positive 6-month (126d) return, inverse-vol weighted; when no asset "
    "passes absolute momentum test hold IEF; monthly rebalance. Conditional "
    "cross-asset risk parity distinct from always-invested risk parity."
)

UNIVERSE = ["QQQ", "GLD", "TLT", "SPY", "IEF"]

STRATEGY = MultiassetAbsMomentumRP()
