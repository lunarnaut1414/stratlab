"""GDX-vs-GLD momentum signal timing QQQ/TLT allocation.

Hypothesis: When gold miners (GDX) outperform physical gold (GLD) on a
42-day basis, it signals risk appetite and broad commodity/equity strength —
miners take on operational risk; they outperform physical gold when investors
are optimistic about economic growth. Use this as a regime gate to hold QQQ
(aggressive) vs TLT (defensive).

This is mechanically distinct from all SP500 stock-picking strategies:
  - Holds QQQ or TLT (ETFs), NOT individual SP500 stocks
  - Signal is GDX/GLD relationship (gold market signal), not equity breadth or credit
  - Not a VIX gate, credit spread gate, or yield curve gate
  - Was already tried for gating SP500 stocks (gen10_gdx_gld_sp500_gate), but
    routing signal to QQQ+TLT allocation is different and potentially lower-corr

Additionally: combine with SPY realized vol for dynamic sizing — reduce QQQ
exposure when SPY 21d vol is elevated (avoid high-vol drawdowns in QQQ).

Design:
  - GDX signal: GDX 42d return vs GLD 42d return. When GDX leads by > 0: risk-on.
  - Risk-on + SPY in uptrend + low vol: QQQ at vol-targeted exposure (clipped 50-97%)
  - Risk-on + SPY in uptrend + high vol: QQQ at 60%
  - Risk-off OR SPY bearish: TLT at 70%
  - SPY 200d SMA outer bear gate
  - Rebalance every 10 bars (biweekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
GDX_GLD_WINDOW = 42      # GDX vs GLD momentum window
SPY_TREND_WINDOW = 200    # outer trend gate
SPY_VOL_WINDOW = 21       # for vol-adaptive QQQ sizing
VOL_TARGET = 0.18         # 18% ann vol target for QQQ
ANNUAL_FACTOR = 252.0
EXPOSURE_HIGH = 0.97
EXPOSURE_LOW = 0.60
DEFENSIVE_WEIGHT = 0.70   # TLT weight in defensive mode


class GDXGLDQQQTiming(Strategy):
    """GDX-vs-GLD 42d signal gates QQQ/TLT allocation; SPY 200d outer gate;
    SPY vol-adaptive QQQ sizing; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        gdx_gld_window: int = GDX_GLD_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        spy_vol_window: int = SPY_VOL_WINDOW,
        vol_target: float = VOL_TARGET,
        exposure_high: float = EXPOSURE_HIGH,
        exposure_low: float = EXPOSURE_LOW,
        defensive_weight: float = DEFENSIVE_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            gdx_gld_window=gdx_gld_window,
            spy_trend_window=spy_trend_window,
            spy_vol_window=spy_vol_window,
            vol_target=vol_target,
            exposure_high=exposure_high,
            exposure_low=exposure_low,
            defensive_weight=defensive_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.gdx_gld_window = int(gdx_gld_window)
        self.spy_trend_window = int(spy_trend_window)
        self.spy_vol_window = int(spy_vol_window)
        self.vol_target = float(vol_target)
        self.exposure_high = float(exposure_high)
        self.exposure_low = float(exposure_low)
        self.defensive_weight = float(defensive_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.gdx_gld_window, self.spy_trend_window, self.spy_vol_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d gate + vol measurement
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + self.spy_vol_window + 5:
            return []
        spy_arr = spy_close.values
        spy_sma = float(np.mean(spy_arr[-self.spy_trend_window:]))
        spy_bull = float(spy_arr[-1]) > spy_sma

        # SPY 21d realized vol
        spy_tail = spy_arr[-(self.spy_vol_window + 1):]
        spy_logr = np.log(spy_tail[1:] / spy_tail[:-1])
        spy_rv = float(np.std(spy_logr))
        spy_ann_vol = spy_rv * (ANNUAL_FACTOR ** 0.5)

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in closes_now.index:
                target["TLT"] = self.defensive_weight
        else:
            # GDX vs GLD signal
            try:
                gdx_hist = ctx.history("GDX")
            except KeyError:
                gdx_hist = None
            try:
                gld_hist = ctx.history("GLD")
            except KeyError:
                gld_hist = None

            risk_on = False
            if gdx_hist is not None and gld_hist is not None:
                gdx_close = gdx_hist["close"].dropna()
                gld_close = gld_hist["close"].dropna()
                if (len(gdx_close) >= self.gdx_gld_window + 2 and
                        len(gld_close) >= self.gdx_gld_window + 2):
                    gdx_ret = float(gdx_close.iloc[-1]) / float(gdx_close.iloc[-self.gdx_gld_window]) - 1.0
                    gld_ret = float(gld_close.iloc[-1]) / float(gld_close.iloc[-self.gdx_gld_window]) - 1.0
                    risk_on = gdx_ret > gld_ret
            else:
                # No GDX/GLD data — default to risk-on (SPY already bullish)
                risk_on = True

            if risk_on:
                # Vol-adaptive QQQ exposure
                if spy_ann_vol > 1e-6:
                    qqq_scale = self.vol_target / spy_ann_vol
                    qqq_exposure = float(np.clip(qqq_scale * self.exposure_high,
                                                  self.exposure_low, self.exposure_high))
                else:
                    qqq_exposure = self.exposure_high
                if "QQQ" in closes_now.index:
                    target["QQQ"] = qqq_exposure
            else:
                # GDX underperforming GLD = cautious — hold TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.defensive_weight

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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


NAME = "gdx_gld_qqq_timing"
HYPOTHESIS = (
    "GDX-vs-GLD 42d return spread gates QQQ/TLT allocation: when GDX outperforms GLD "
    "(miners leading physical gold = risk appetite signal) hold QQQ sized to 18% ann vol "
    "target via SPY 21d realized vol; when GLD leads GDX hold TLT 70%; SPY 200d outer bear "
    "gate forces TLT regardless; biweekly rebalance — gold-miners-vs-gold signal not used "
    "for ETF allocation on leaderboard; holds ETFs not individual stocks"
)

UNIVERSE = ["GDX", "GLD", "QQQ", "TLT", "SPY"]

STRATEGY = GDXGLDQQQTiming()
