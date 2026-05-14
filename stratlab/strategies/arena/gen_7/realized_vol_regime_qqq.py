"""SPY realized-volatility ratio regime: QQQ/SPY/TLT tiered allocation.

Hypothesis: When SPY 10d realized vol is below its 63d rolling average (calm
regime), hold QQQ 97%. When 10d vol is between the 63d average and 1.5x that
average (moderate regime), hold SPY 97%. When 10d vol exceeds 1.5x the 63d
average (stress regime), hold TLT 60% + SHY 37%. Weekly rebalance.

Rationale:
  - Realized volatility (computed from daily returns) captures actual risk, not
    market expectations like VIX. The ratio of short-term to long-term realized
    vol is self-normalizing: it adapts to secular vol shifts rather than using
    absolute thresholds that may be regime-specific.
  - In 2010-2018, typical SPY 63d realized vol ranged from 7-18%. A fixed VIX
    threshold of 25 may miss stress episodes in low-vol environments. The 10d/63d
    ratio provides a relative measure that works across different vol levels.
  - QQQ in calm regime captures the tech momentum premium of low-vol bull markets.
  - Tiering to SPY (not immediately to TLT) in moderate vol avoids over-rotation.
  - The realized vol signal is orthogonal to:
    * VIX level strategies (implied vs realized vol)
    * Credit spread strategies (bond market vs equity vol)
    * Breadth strategies (individual stock performance vs market vol)
    * Price momentum strategies (returns-based vs vol-based signal)

Distinction from existing strategies:
  - VIX-gated strategies use ^VIX as implied vol threshold (15, 20, 22, 25, etc.)
  - This uses SPY REALIZED vol ratio (10d/63d average) as self-normalizing signal
  - No dependence on VIX data (signal computed purely from SPY price history)
  - The 1.5x threshold creates regime sensitivity proportional to current vol level
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "TLT", "SHY"]

REBALANCE_EVERY = 5        # weekly
SHORT_VOL = 10             # 10-day realized vol window
LONG_VOL = 63              # 63-day rolling average window
STRESS_RATIO = 1.5         # 10d vol > 1.5x 63d avg = stress
EXPOSURE = 0.97
CALM_ETF = "QQQ"           # hold QQQ in calm regime
MODERATE_ETF = "SPY"       # hold SPY in moderate regime
STRESS_BOND_ETF = "TLT"    # hold TLT in stress regime
STRESS_CASH_ETF = "SHY"    # hold SHY in stress regime
STRESS_BOND_WEIGHT = 0.60
STRESS_CASH_WEIGHT = 0.37  # 0.60 + 0.37 = 0.97


class RealizedVolRegimeQQQ(Strategy):
    """SPY realized-vol ratio regime: QQQ (calm) / SPY (moderate) / TLT+SHY (stress)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_vol: int = SHORT_VOL,
        long_vol: int = LONG_VOL,
        stress_ratio: float = STRESS_RATIO,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_vol=short_vol,
            long_vol=long_vol,
            stress_ratio=stress_ratio,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_vol = int(short_vol)
        self.long_vol = int(long_vol)
        self.stress_ratio = float(stress_ratio)
        self.exposure = float(exposure)

    def _realized_vol(self, prices: "np.ndarray", window: int) -> float:
        """Annualized realized vol over the given window of prices."""
        if len(prices) < window + 1:
            return float("nan")
        tail = prices[-(window + 1):]
        logr = np.log(tail[1:] / tail[:-1])
        rv = float(np.std(logr)) * np.sqrt(252)
        return rv

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.long_vol + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get SPY history for realized vol computation
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.long_vol + 5:
            return []
        spy_close = spy_hist["close"].dropna().values
        if len(spy_close) < self.long_vol + 1:
            return []

        # 10d realized vol
        rv_short = self._realized_vol(spy_close, self.short_vol)
        if not np.isfinite(rv_short):
            return []

        # 63d average of rolling 10d realized vol (compute rolling RV over long window)
        # Compute the 10d realized vol at each of the last long_vol bars
        rv_series = []
        for i in range(self.long_vol):
            end_idx = len(spy_close) - i
            if end_idx < self.short_vol + 1:
                break
            tail = spy_close[end_idx - self.short_vol - 1: end_idx]
            if len(tail) < self.short_vol + 1:
                break
            logr = np.log(tail[1:] / tail[:-1])
            rv = float(np.std(logr)) * np.sqrt(252)
            if np.isfinite(rv):
                rv_series.append(rv)

        if len(rv_series) < self.long_vol // 2:
            return []

        rv_long_avg = float(np.mean(rv_series))
        if rv_long_avg <= 0 or not np.isfinite(rv_long_avg):
            return []

        ratio = rv_short / rv_long_avg

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine regime
        target: dict[str, float] = {}
        if ratio <= 1.0:
            # Calm regime: QQQ
            if CALM_ETF in closes_now.index:
                target[CALM_ETF] = self.exposure
        elif ratio <= self.stress_ratio:
            # Moderate regime: SPY
            if MODERATE_ETF in closes_now.index:
                target[MODERATE_ETF] = self.exposure
        else:
            # Stress regime: TLT + SHY
            if STRESS_BOND_ETF in closes_now.index:
                target[STRESS_BOND_ETF] = STRESS_BOND_WEIGHT
            if STRESS_CASH_ETF in closes_now.index:
                target[STRESS_CASH_ETF] = STRESS_CASH_WEIGHT

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


NAME = "realized_vol_regime_qqq"
HYPOTHESIS = (
    "SPY realized-vol ratio regime: when SPY 10d realized vol below 63d average hold QQQ 97%; "
    "when 10d vol 63d-to-1.5x hold SPY 97%; when 10d vol above 1.5x average hold TLT 60%+SHY 37%; "
    "weekly rebalance; uses actual realized volatility ratio as self-normalizing regime signal "
    "distinct from VIX implied-vol level"
)

STRATEGY = RealizedVolRegimeQQQ()
