"""VIX-regime safe-haven rotation: GLD vs TLT vs SPY tri-state allocator.

Hypothesis:
  Three-state regime using VIX level and SPY 200d SMA:
    State 1 (calm bull):   VIX <= 18 AND SPY above 200d SMA
                           → SPY 90% (max equity, low fear)
    State 2 (fear spike):  VIX > 18 AND VIX 10d MA > 1.2x VIX 60d MA
                           → GLD 50% + TLT 47% (flight to safety,
                             gold+bonds as dual safe-haven)
    State 3 (bear market): SPY below 200d SMA (structural downtrend)
                           → TLT 90% (duration bet in risk-off)
    Default (transitional): SPY 60% + GLD 37%

  The key insight: during a fear spike (VIX acceleration), gold and bonds
  together outperform pure equity or pure bonds. Gold benefits from
  uncertainty/USD weakness; bonds benefit from flight-to-safety and rate
  cuts. TLT-only defensive misses gold in scenarios where rates rise despite
  fear (e.g. credit crises). Rebalance weekly (every 5 bars).

Rationale:
  Existing leaderboard strategies use VIX as a binary gate (hold vs don't hold
  equities). This strategy uses VIX acceleration (ratio of short-MA to long-MA)
  as a FEAR SPIKE signal, distinct from level thresholds. The dual safe-haven
  allocation (GLD + TLT) is not used by any existing strategy.

  During 2010-2018 IS window:
  - 2011 US debt ceiling + EU sovereign crisis: VIX spiked → GLD+TLT hold
  - 2015-2016 China fears: VIX spiked → GLD+TLT hold
  - Calm periods (most of 2012-2015): SPY hold

Diversification vs leaderboard:
  - VIX-gated strategies: binary on/off; this uses VIX acceleration + level
  - Risk parity: always holds all 3; this is regime-conditional
  - Bond-equity regime (TLT/SPY ratio): uses ratio MA, this uses VIX accel
  - Halloween seasonal: calendar-based, this is signal-based
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

VIX_CALM_THRESHOLD = 18.0    # VIX below this = calm regime
VIX_ACCEL_RATIO = 1.2        # short_MA / long_MA > this = fear spike
VIX_SHORT_MA = 10            # short VIX MA for acceleration
VIX_LONG_MA = 60             # long VIX MA for acceleration baseline
TREND_WINDOW = 200           # SPY 200d SMA
REBALANCE_EVERY = 5          # weekly

# Exposure levels
BULL_SPY_EXPOSURE = 0.90     # calm bull: SPY
FEAR_GLD_EXPOSURE = 0.50     # fear spike: GLD
FEAR_TLT_EXPOSURE = 0.47     # fear spike: TLT
BEAR_TLT_EXPOSURE = 0.90     # bear market: TLT
DEFAULT_SPY = 0.60           # transitional: SPY
DEFAULT_GLD = 0.37           # transitional: GLD

_VIX = "^VIX"


class VixSafehavenGldTltSpy(Strategy):
    """Three-state VIX-regime allocator: SPY/GLD+TLT/TLT."""

    def __init__(
        self,
        vix_calm: float = VIX_CALM_THRESHOLD,
        vix_accel_ratio: float = VIX_ACCEL_RATIO,
        vix_short_ma: int = VIX_SHORT_MA,
        vix_long_ma: int = VIX_LONG_MA,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        bull_spy_exp: float = BULL_SPY_EXPOSURE,
        fear_gld_exp: float = FEAR_GLD_EXPOSURE,
        fear_tlt_exp: float = FEAR_TLT_EXPOSURE,
        bear_tlt_exp: float = BEAR_TLT_EXPOSURE,
        default_spy: float = DEFAULT_SPY,
        default_gld: float = DEFAULT_GLD,
    ) -> None:
        super().__init__(
            vix_calm=vix_calm,
            vix_accel_ratio=vix_accel_ratio,
            vix_short_ma=vix_short_ma,
            vix_long_ma=vix_long_ma,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
        )
        self.vix_calm = float(vix_calm)
        self.vix_accel_ratio = float(vix_accel_ratio)
        self.vix_short_ma = int(vix_short_ma)
        self.vix_long_ma = int(vix_long_ma)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.bull_spy_exp = float(bull_spy_exp)
        self.fear_gld_exp = float(fear_gld_exp)
        self.fear_tlt_exp = float(fear_tlt_exp)
        self.bear_tlt_exp = float(bear_tlt_exp)
        self.default_spy = float(default_spy)
        self.default_gld = float(default_gld)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vix_long_ma + 5
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

        # --- SPY 200d SMA ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull_market = float(spy_close.iloc[-1]) > spy_sma

        # --- VIX signals ---
        vix_level = float("nan")
        fear_spike = False
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= self.vix_long_ma + 1:
                vix_close = vix_hist["close"].dropna()
                vix_level = float(vix_close.iloc[-1])
                short_ma = float(vix_close.iloc[-self.vix_short_ma:].mean())
                long_ma = float(vix_close.iloc[-self.vix_long_ma:].mean())
                if long_ma > 0:
                    fear_spike = (short_ma / long_ma) > self.vix_accel_ratio
        except Exception:
            pass

        # --- Regime determination ---
        target: dict[str, float] = {}

        calm_vix = np.isfinite(vix_level) and vix_level <= self.vix_calm
        high_vix = np.isfinite(vix_level) and vix_level > self.vix_calm

        if bull_market and calm_vix:
            # State 1: calm bull - max equity
            if "SPY" in closes_now.index:
                target["SPY"] = self.bull_spy_exp
        elif fear_spike and high_vix:
            # State 2: VIX acceleration spike - dual safe haven
            if "GLD" in closes_now.index:
                target["GLD"] = self.fear_gld_exp
            if "TLT" in closes_now.index:
                target["TLT"] = self.fear_tlt_exp
        elif not bull_market:
            # State 3: bear market structural downtrend
            if "TLT" in closes_now.index:
                target["TLT"] = self.bear_tlt_exp
        else:
            # Default: transitional / mild elevated VIX in bull market
            if "SPY" in closes_now.index:
                target["SPY"] = self.default_spy
            if "GLD" in closes_now.index:
                target["GLD"] = self.default_gld

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


NAME = "vix_safehaven_gld_tlt_spy"
HYPOTHESIS = (
    "VIX-regime safe-haven tri-state: SPY 90% when VIX<=18 and bull market; "
    "GLD 50%+TLT 47% when VIX 10d MA/60d MA > 1.2 (fear spike); TLT 90% in "
    "bear market (SPY<200d SMA); SPY 60%+GLD 37% default. Weekly rebalance."
)

UNIVERSE = ["SPY", "GLD", "TLT", _VIX]

STRATEGY = VixSafehavenGldTltSpy()
