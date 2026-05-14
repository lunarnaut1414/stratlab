"""Realized volatility carry on SPY strategy.

Hypothesis: When SPY 21d realized vol is below its 90d median (vol carry positive,
calm regime), increase equity allocation to 90%; scale back to 50% SPY + 47% TLT
when realized vol is above 90d median (vol expansion, carry negative). Weekly
rebalance on SPY only.

Rationale: Realized volatility regimes persist — low-vol environments tend to
persist until a shock. When 21d RV < 90d median, the market is in a calm phase
where equity carry is positive. This is distinct from VIX level (implied vol)
and uses only backward-looking realized vol. It avoids the binary on/off nature
of VIX thresholds.

Distinction from existing strategies:
  - Uses realized vol (not VIX level) vs its own trailing median as signal
  - Three-tier allocation: high/medium/low vol -> 90%/70%/50% SPY
  - Pure SPY universe — no stock picking, avoids corr issues with momentum
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
RV_WINDOW = 21          # 21d realized vol (1 month)
MEDIAN_WINDOW = 90      # 90d window for median comparison
EXPOSURE_HIGH = 0.90    # low vol (calm) -> high equity
EXPOSURE_MID = 0.70     # mid vol
EXPOSURE_LOW = 0.50     # high vol -> reduce equity, add bonds


class RealizedVolCarrySpy(Strategy):
    """SPY allocation scaled by 21d RV vs 90d median: calm->90%, rising->70%, stressed->50%+TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rv_window: int = RV_WINDOW,
        median_window: int = MEDIAN_WINDOW,
        exposure_high: float = EXPOSURE_HIGH,
        exposure_mid: float = EXPOSURE_MID,
        exposure_low: float = EXPOSURE_LOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rv_window=rv_window,
            median_window=median_window,
            exposure_high=exposure_high,
            exposure_mid=exposure_mid,
            exposure_low=exposure_low,
        )
        self.rebalance_every = int(rebalance_every)
        self.rv_window = int(rv_window)
        self.median_window = int(median_window)
        self.exposure_high = float(exposure_high)
        self.exposure_mid = float(exposure_mid)
        self.exposure_low = float(exposure_low)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.median_window + self.rv_window + 10
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

        # Compute SPY realized vol and its rolling median
        spy_exposure = self.exposure_mid  # default mid
        tlt_exposure = 0.0
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.median_window + self.rv_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.median_window + self.rv_window:
                    # Compute rolling 21d RV series over the last 90d
                    log_rets = np.log(spy_close.values[1:] / spy_close.values[:-1])

                    # Current 21d RV (annualized)
                    if len(log_rets) >= self.rv_window:
                        current_rv = float(np.std(log_rets[-self.rv_window:]) * np.sqrt(252))
                    else:
                        current_rv = float("nan")

                    # Compute rolling 21d RV for each day in the last median_window days
                    rv_series = []
                    for i in range(self.median_window):
                        end_i = len(log_rets) - i
                        start_i = end_i - self.rv_window
                        if start_i < 0:
                            break
                        rv_i = float(np.std(log_rets[start_i:end_i]) * np.sqrt(252))
                        rv_series.append(rv_i)

                    if rv_series and np.isfinite(current_rv):
                        median_rv = float(np.median(rv_series))
                        # Regime: below 33rd percentile = calm, above 67th = stressed
                        p33 = float(np.percentile(rv_series, 33))
                        p67 = float(np.percentile(rv_series, 67))

                        if current_rv <= p33:
                            # Calm: high equity allocation
                            spy_exposure = self.exposure_high
                            tlt_exposure = 0.0
                        elif current_rv >= p67:
                            # Stressed: low equity + bonds
                            spy_exposure = self.exposure_low
                            tlt_exposure = 0.47
                        else:
                            # Middle: medium equity
                            spy_exposure = self.exposure_mid
                            tlt_exposure = 0.0
        except Exception:
            pass

        target: dict[str, float] = {}
        if "SPY" in closes_now.index:
            target["SPY"] = spy_exposure
        if tlt_exposure > 0 and "TLT" in closes_now.index:
            target["TLT"] = tlt_exposure

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


NAME = "realized_vol_carry_spy"
HYPOTHESIS = (
    "Realized vol carry SPY: when SPY 21d realized vol is below 33rd percentile of 90d RV "
    "distribution (calm regime) hold SPY at 90%; above 67th pct (stressed) hold SPY 50%+TLT 47%; "
    "middle regime hold SPY 70%; weekly rebalance; pure realized-vol-carry angle not VIX-level gate"
)

UNIVERSE = ["SPY", "TLT"]

STRATEGY = RealizedVolCarrySpy()
