"""Risk Parity with Small-Cap Credit Tilt — gen_8 sonnet-4

Hypothesis: Always-invested strategy using inverse-vol weighted baseline
of SPY/IEF/GLD (standard risk parity). When JNK is above its 21d MA
(credit risk-on), divert 20% of SPY's base allocation to IWM
(small-cap tilt — riskier, higher expected return in credit expansion).
When JNK below 21d MA (credit stress), divert 20% of SPY's allocation to
TLT (flight to safety within the risk-parity framework). Monthly rebalance.

Rationale: Risk parity is always invested, providing returns in bull IS window.
The credit tilt injects small-cap alpha when credit conditions are favorable
(JNK > 21d MA): 2010-2014 credit expansion was particularly strong for
IWM outperformance. The defensive shift from SPY to TLT when credit deteriorates
reduces drawdowns vs pure risk parity.

Distinction from existing strategies:
  - gen6_rp_credit_tilt: uses JNK 30d MA, tilts SPY allocation up/down within
    SPY itself (no IWM). This uses JNK 21d MA, diverts to IWM for credit-on
    (different risk-on asset), and to TLT for credit-off (reduces equity further).
  - gen5_risk_parity_spy_tlt_gld: pure RP without credit signal.
  - This adds a 3-way branching credit tilt (credit-on→IWM, neutral→RP, credit-off→TLT)
    vs the binary tilt of gen6_rp_credit_tilt.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21       # monthly
VOL_WINDOW = 20            # realized vol for inverse-vol sizing
JNK_MA = 21                # credit MA window
CREDIT_TILT = 0.20         # fraction of SPY to shift to IWM or TLT
EXPOSURE = 0.97


class RpSmallcapCreditTilt(Strategy):
    """Always-invested risk parity SPY/IEF/GLD with JNK-driven IWM tilt."""

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
        warmup = max(self.vol_window, self.jnk_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute inverse-vol for SPY, IEF, GLD base assets
        base_assets = ["SPY", "IEF", "GLD"]
        need = self.vol_window + 5
        prices = ctx.closes_window(need)

        inv_vols: dict[str, float] = {}
        for sym in base_assets:
            if sym not in prices.columns or sym not in live:
                continue
            col = prices[sym].dropna()
            if len(col) < self.vol_window + 1:
                continue
            daily_rets = col.pct_change().dropna()
            if len(daily_rets) < 5:
                continue
            vol = float(daily_rets.iloc[-self.vol_window:].std())
            if vol > 0:
                inv_vols[sym] = 1.0 / vol

        if not inv_vols:
            return []

        total_inv_vol = sum(inv_vols.values())

        # Base risk-parity weights
        base_weights = {sym: inv_vols[sym] / total_inv_vol for sym in inv_vols}

        # JNK credit signal
        credit_on = False
        credit_off = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    credit_on = jnk_now > jnk_ma_val
                    credit_off = jnk_now < jnk_ma_val * 0.99  # slight buffer
        except Exception:
            pass

        # Apply credit tilt
        final_weights: dict[str, float] = {}

        spy_base = base_weights.get("SPY", 0.0)
        ief_base = base_weights.get("IEF", 0.0)
        gld_base = base_weights.get("GLD", 0.0)

        if credit_on and "IWM" in live:
            # Shift CREDIT_TILT fraction of SPY to IWM
            tilt = spy_base * self.credit_tilt
            final_weights["SPY"] = spy_base - tilt
            final_weights["IWM"] = tilt
            final_weights["IEF"] = ief_base
            final_weights["GLD"] = gld_base
        elif credit_off and "TLT" in live:
            # Shift CREDIT_TILT fraction of SPY to TLT (more defensive)
            tilt = spy_base * self.credit_tilt
            final_weights["SPY"] = spy_base - tilt
            final_weights["TLT"] = tilt
            final_weights["IEF"] = ief_base
            final_weights["GLD"] = gld_base
        else:
            # Neutral: plain risk parity
            final_weights.update(base_weights)

        # Apply exposure cap
        target: dict[str, float] = {
            sym: w * self.exposure
            for sym, w in final_weights.items()
            if sym in live and w > 0
        }

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


UNIVERSE = ["SPY", "IEF", "GLD", "TLT", "IWM", "JNK"]

NAME = "rp_smallcap_credit_tilt"
HYPOTHESIS = (
    "Always-invested risk parity with JNK credit tilt: always-invested SPY/IEF/GLD "
    "inverse-vol weighted baseline; when JNK above 21d MA (credit risk-on) add 20% "
    "of SPY weight to IWM (small-cap tilt); when JNK below 21d MA (credit stress) "
    "shift 20% SPY weight to TLT; monthly rebalance; extension of risk-parity with "
    "distinct credit-tilted small-cap angle"
)

STRATEGY = RpSmallcapCreditTilt()
