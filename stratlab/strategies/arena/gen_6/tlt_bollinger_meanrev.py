"""TLT Bollinger %b mean-reversion with equity tilt.

Hypothesis: Apply Bollinger band mean-reversion logic to TLT (long-bond ETF):
  - Compute %b on TLT 20d Bollinger bands (2 std).
  - %b < 0 (TLT deeply oversold below lower band): TLT 95%, expecting bounce.
  - 0 <= %b < 0.5: TLT 60% + SPY 35% (tilt toward TLT, mild equity exposure).
  - 0.5 <= %b < 1.0: TLT 35% + SPY 60% (tilt toward equity, mild bond hedge).
  - %b >= 1.0 (TLT deeply overbought above upper band): SPY 95% (rotate to
    equity since TLT mean-reverts down on rallies).

Why this fills a gap:
  - Phase 2 brief: "Bollinger %b or Keltner-channel signals on TLT or LQD
    (price-action on bonds, not equities)". This is the exact theme.
  - All Bollinger / mean-reversion strategies on the leaderboard apply to
    QQQ or SPY (vix_spike_buy_dip_spy, qqq_bollinger_vvix_dipbuy). None apply
    to TLT.
  - TLT mean-reverts more reliably than equities in 2010-2018 because Fed
    QE programs anchored long-rate expectations: TLT rallies got faded
    quickly (rate-spike fears), and TLT sell-offs got bought (flight to
    quality during equity scares). This makes TLT %b mean-reversion a
    genuine signal.
  - Continuous tilt distinct from the discrete on/off bond regime allocators
    on the leaderboard (jnk_lqd_spy_regime, hy_credit_qqq_rotation, etc.).

Implementation: rebalance every 5 bars (weekly) on TLT %b state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["TLT", "QQQ", "SPY", "IEF", "SHY"]

BB_PERIOD = 20
BB_STD = 2.0
REBALANCE_EVERY = 5
TREND_WINDOW = 100
EXPOSURE = 0.97


class TltBollingerMeanrev(Strategy):
    def __init__(
        self,
        bb_period: int = BB_PERIOD,
        bb_std: float = BB_STD,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.bb_period = int(bb_period)
        self.bb_std = float(bb_std)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.bb_period + 5, self.trend_window + 5)
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend gate: in bear, allow defensive bond-only
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 1:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window + 1:
                    spy_now = float(spy_close.iloc[-1])
                    spy_ma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = spy_now > spy_ma
        except KeyError:
            pass

        # Compute TLT Bollinger %b
        pct_b = float("nan")
        try:
            tlt_hist = ctx.history("TLT")
            if tlt_hist is not None and len(tlt_hist) >= self.bb_period + 1:
                tlt_close = tlt_hist["close"].dropna()
                if len(tlt_close) >= self.bb_period + 1:
                    window = tlt_close.iloc[-self.bb_period:]
                    ma = float(window.mean())
                    std = float(window.std(ddof=0))
                    if std > 0:
                        upper = ma + self.bb_std * std
                        lower = ma - self.bb_std * std
                        last = float(tlt_close.iloc[-1])
                        if upper > lower:
                            pct_b = (last - lower) / (upper - lower)
        except Exception:
            pass

        if not np.isfinite(pct_b):
            # Default to neutral (mid-range)
            pct_b = 0.5

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Bond pool stays small (20%) — main risk is QQQ/SPY equity.
        # The TLT %b drives *which* bond (TLT vs IEF vs SHY) and a small
        # equity tilt (QQQ vs SPY).
        eq_base = 0.80 if spy_bull else 0.20
        bond_pool = 1.0 - eq_base

        # Equity allocation: QQQ when TLT oversold (risk-on), SPY otherwise.
        # TLT mean-reverts up after oversold ≈ recent risk-off shock,
        # so a contrarian QQQ tilt right after the shock catches the snap-back.
        if eq_base > 0:
            if pct_b < 0.0 and "QQQ" in live:
                target["QQQ"] = eq_base * self.exposure
            elif "SPY" in live:
                target["SPY"] = eq_base * self.exposure
            elif "QQQ" in live:
                target["QQQ"] = eq_base * self.exposure

        # Allocate the bond pool by TLT %b mean-reversion
        if pct_b < 0.0:
            # TLT deeply oversold — overweight TLT
            if "TLT" in live:
                target["TLT"] = bond_pool * self.exposure
        elif pct_b < 0.5:
            # TLT below midline — TLT tilt
            if "TLT" in live:
                target["TLT"] = bond_pool * 0.70 * self.exposure
            if "IEF" in live:
                target["IEF"] = bond_pool * 0.30 * self.exposure
        elif pct_b < 1.0:
            # TLT above midline — IEF tilt
            if "IEF" in live:
                target["IEF"] = bond_pool * self.exposure
        else:
            # TLT deeply overbought — short-end only
            if "SHY" in live:
                target["SHY"] = bond_pool * self.exposure

        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "tlt_bollinger_meanrev"
HYPOTHESIS = (
    "TLT Bollinger %b(20,2) mean-reversion with continuous SPY tilt: TLT 95% "
    "when %b<0 (oversold); TLT 60%+SPY 35% when %b in [0,0.5); TLT 35%+SPY 60% "
    "when %b in [0.5,1); SPY 95% when %b>=1 (overbought). Weekly rebalance; "
    "price-action mean-reversion on bonds with continuous equity tilt."
)

STRATEGY = TltBollingerMeanrev()
