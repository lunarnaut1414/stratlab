"""GLD/TLT inflation-regime rotation with SPY bull overlay.

Hypothesis: The divergence between gold (GLD) and treasury bonds (TLT) signals
inflation expectations. When gold is in an uptrend AND bonds are weak,
inflation is rising and gold outperforms. When bonds are strong AND gold is
weak, deflationary/growth slowdown concerns dominate. When both are positive
(low-vol expansion), equities win.

Signal logic (weekly rebalance):
  - GLD golden cross (50d > 200d) AND TLT 20d return < 0 (bonds weak):
    INFLATION REGIME -> hold GLD 97%
  - TLT 20d return > 0 (bonds strengthening) AND GLD 50d < 200d (gold falling):
    DEFLATION/SAFETY REGIME -> hold TLT 97%
  - GLD 50d > 200d AND TLT 20d return > 0 (both bullish):
    GOLDILOCKS -> hold SPY 97%
  - Default/SPY bear (SPY < 150d SMA): hold TLT 97%

Rationale: Gold and bonds move in opposite directions in inflation vs
deflation. Their joint trend provides a three-way macro regime signal that
is orthogonal to VIX (fear), credit spreads (stress), and equity breadth.
This is similar to the curated gen5_rp_credit_tilt but uses gold-bond
divergence not credit spreads.

Distinct from:
  - All VIX/credit/breadth regime strategies
  - gen5_risk_parity (equal-weight risk parity, not directional timing)
  - Any strategy not using GLD/TLT divergence as primary signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
GLD_FAST_MA = 50           # GLD fast SMA for golden cross
GLD_SLOW_MA = 200          # GLD slow SMA for golden cross
TLT_SHORT_WINDOW = 20      # TLT 20d return signal
SPY_TREND_WINDOW = 150     # SPY bear market gate
EXPOSURE = 0.97

_GLD = "GLD"
_TLT = "TLT"
_SPY = "SPY"


class GLDTLTInflationRegime(Strategy):
    """GLD/TLT inflation-deflation-goldilocks tri-state regime allocator.

    Uses gold golden cross + bond short-term return direction to classify
    the macro regime and route to GLD, TLT, or SPY accordingly.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        gld_fast_ma: int = GLD_FAST_MA,
        gld_slow_ma: int = GLD_SLOW_MA,
        tlt_short_window: int = TLT_SHORT_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            gld_fast_ma=gld_fast_ma,
            gld_slow_ma=gld_slow_ma,
            tlt_short_window=tlt_short_window,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.gld_fast_ma = int(gld_fast_ma)
        self.gld_slow_ma = int(gld_slow_ma)
        self.tlt_short_window = int(tlt_short_window)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.gld_slow_ma, self.spy_trend_window) + 10
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

        # SPY bear market gate
        spy_bull = False
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.spy_trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.spy_trend_window:
                    spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # GLD golden cross signal (50d SMA vs 200d SMA)
        gld_golden_cross = False
        try:
            gld_hist = ctx.history(_GLD)
            if gld_hist is not None and len(gld_hist) >= self.gld_slow_ma:
                gld_close = gld_hist["close"].dropna()
                if len(gld_close) >= self.gld_slow_ma:
                    gld_fast = float(gld_close.iloc[-self.gld_fast_ma:].mean())
                    gld_slow = float(gld_close.iloc[-self.gld_slow_ma:].mean())
                    gld_golden_cross = gld_fast > gld_slow
        except Exception:
            pass

        # TLT 20d return signal
        tlt_positive = False
        try:
            tlt_hist = ctx.history(_TLT)
            if tlt_hist is not None and len(tlt_hist) >= self.tlt_short_window + 1:
                tlt_close = tlt_hist["close"].dropna()
                if len(tlt_close) >= self.tlt_short_window + 1:
                    tlt_ret = float(
                        tlt_close.iloc[-1] / tlt_close.iloc[-(self.tlt_short_window + 1)] - 1.0
                    )
                    tlt_positive = np.isfinite(tlt_ret) and tlt_ret > 0
        except Exception:
            pass

        # Determine allocation
        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — hold TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        elif gld_golden_cross and not tlt_positive:
            # Inflation regime: gold up, bonds weak -> GLD
            if "GLD" in live:
                target["GLD"] = self.exposure
        elif tlt_positive and not gld_golden_cross:
            # Deflation/safety regime: bonds up, gold down -> TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        elif gld_golden_cross and tlt_positive:
            # Goldilocks: both bullish -> SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Neither bullish (gold falling, bonds weak) -> TLT defensive
            if "TLT" in live:
                target["TLT"] = self.exposure

        # Build orders
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


NAME = "gld_tlt_inflation_regime"
HYPOTHESIS = (
    "GLD/TLT inflation-regime rotation: hold GLD 97% when GLD golden cross (50d>200d) AND "
    "TLT 20d return negative (inflation regime); hold TLT when bonds strengthening AND gold "
    "downtrend; hold SPY when both bullish (goldilocks); TLT default; weekly rebalance; "
    "gold-bond divergence as inflation signal orthogonal to VIX/credit/breadth"
)

UNIVERSE = ["GLD", "TLT", "SPY"]

STRATEGY = GLDTLTInflationRegime()
