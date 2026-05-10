"""Risk parity with JNK credit tilt — gen_6 sonnet-7

Hypothesis: Base allocation is SPY/TLT/GLD inverse-vol weighted (20d realized vol).
Apply a credit-driven SPY tilt: when JNK is above its 30d SMA (credit healthy),
increase SPY weight by 50% of its inverse-vol share (reallocated from TLT and GLD).
When JNK is below its 30d SMA, use pure inverse-vol weights.
Always hold SPY/TLT/GLD, never go to cash. Rebalance every 10 bars.

Rationale:
  Pure risk parity (gen5_risk_parity_spy_tlt_gld) has IS Calmar 0.62 and low corr.
  Adding a credit tilt that overweights SPY when credit is healthy captures the
  risk-on premium while maintaining the diversification benefit of the base parity
  allocation. Different from credit-switching strategies (which go all-in one asset)
  because the allocation always has all three assets — it just shifts the tilt.

  Distinct from existing strategies:
  - Always holds SPY+TLT+GLD (no full rotation to any single asset)
  - Credit tilt within a risk parity framework (not credit gating)
  - Different from gen6_risk_parity_4asset_vnq (uses SPY/TLT/GLD only, no VNQ,
    and adds credit tilt which VNQ version doesn't have)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
VOL_WINDOW = 20         # 20d realized vol for weights
JNK_MA = 30             # JNK 30d SMA for credit signal
CREDIT_TILT = 0.20      # additional SPY weight fraction when credit healthy
EXPOSURE = 0.97
ASSETS = ["SPY", "TLT", "GLD"]


class RPCreditTilt(Strategy):
    """Risk parity with JNK credit tilt on SPY weight."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        jnk_ma: int = JNK_MA,
        credit_tilt: float = CREDIT_TILT,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            jnk_ma=jnk_ma,
            credit_tilt=credit_tilt,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.jnk_ma = int(jnk_ma)
        self.credit_tilt = float(credit_tilt)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vol_window, self.jnk_ma) + 5
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

        # JNK credit signal
        jnk_healthy = True
        try:
            jnk_hist = ctx.history("JNK")
            if len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna().values
                jnk_sma = float(np.mean(jnk_close[-self.jnk_ma:]))
                jnk_healthy = float(jnk_close[-1]) > jnk_sma
        except Exception:
            pass

        # Compute inverse-vol weights for SPY/TLT/GLD
        inv_vols: dict[str, float] = {}
        for sym in ASSETS:
            if sym not in closes_now.index:
                continue
            try:
                hist = ctx.history(sym)
            except Exception:
                continue
            if len(hist) < self.vol_window + 2:
                continue
            close_arr = hist["close"].dropna().values
            if len(close_arr) < self.vol_window + 1:
                continue
            logr = np.log(close_arr[-self.vol_window - 1:][1:] / close_arr[-self.vol_window - 1:][:-1])
            rv = float(np.std(logr))
            if rv > 1e-6 and np.isfinite(rv):
                inv_vols[sym] = 1.0 / rv

        if not inv_vols or "SPY" not in inv_vols:
            # Can't compute weights, hold SPY
            target: dict[str, float] = {"SPY": self.exposure}
        else:
            iv_sum = sum(inv_vols.values())
            if iv_sum <= 0:
                target = {"SPY": self.exposure}
            else:
                # Base risk parity weights
                weights: dict[str, float] = {s: self.exposure * iv / iv_sum for s, iv in inv_vols.items()}

                # Apply credit tilt: increase SPY, reduce TLT/GLD proportionally
                if jnk_healthy and "SPY" in weights:
                    tilt_amount = self.credit_tilt * weights["SPY"]  # add 20% of SPY's RP weight to SPY
                    # Take from TLT and GLD proportionally
                    non_spy_sum = sum(w for s, w in weights.items() if s != "SPY")
                    if non_spy_sum > 0:
                        for sym in list(weights.keys()):
                            if sym != "SPY":
                                weights[sym] -= tilt_amount * weights[sym] / non_spy_sum
                        weights["SPY"] += tilt_amount

                target = {s: max(0.01, w) for s, w in weights.items()}

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


NAME = "rp_credit_tilt"
HYPOTHESIS = (
    "Risk parity SPY/TLT/GLD with JNK 30d SMA credit tilt: inverse-vol weighted always-invested; "
    "when JNK>30d SMA increase SPY weight by 20% of its RP share (reduce TLT+GLD); "
    "biweekly rebalance; base RP + credit-driven equity overweight inside parity framework"
)
UNIVERSE = ["SPY", "TLT", "GLD", "JNK"]
STRATEGY = RPCreditTilt()
