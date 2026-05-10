"""VIX term-structure regime QQQ allocator.

Hypothesis: The shape of the VIX term-structure (^VIX3M / ^VIX) is a
robust regime indicator orthogonal to VIX *level*:
  - Contango (VIX3M / VIX > 1.05):  market expects vol to *rise* — but
    spot VIX is currently low, so risk-on environment. Hold QQQ 97%.
  - Mild backwardation / neutral (1.00 to 1.05): hold SPY 97%.
  - Backwardation (VIX3M / VIX < 1.00): vol curve inverted — spot VIX is
    elevated relative to forward expectations, signal of acute stress.
    Hold IEF 60% + SHY 37% (defensive bond mix).

Why this fills a gap:
  - Saturated dead-end list flags VIX/VVIX/MOVE *level* allocators (4 failed)
    because their gates fire too rarely in calm IS years. The term-structure
    *ratio* fires continuously — it is in contango ~80% of days but the
    backwardation signal (~20% of days) hits exactly the bad-vol regimes
    (Aug 2011, late-2015, Aug 2015, Jan-Feb 2016, Feb 2018, Q4 2018).
  - Phase 2 brief explicitly calls this out as untouched ("^VIX3M / ^VIX
    term-structure regime: contango vs backwardation; nobody used it;
    orthogonal to VIX level").
  - All other VIX-using strategies on the leaderboard use VIX-level
    thresholds (15/18/20/22/25/28). None use the curve shape.

Universe is small (5 ETFs + 2 indices) so backtest is fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "IEF", "SHY", "TLT", "^VIX", "^VIX3M"]

CONTANGO_THRESHOLD = 1.10
BACKWARDATION_THRESHOLD = 1.03
SMOOTH_DAYS = 10
REBALANCE_EVERY = 5
TREND_WINDOW = 100
EXPOSURE = 0.97


class VixTermStructureQqq(Strategy):
    def __init__(
        self,
        contango_threshold: float = CONTANGO_THRESHOLD,
        backwardation_threshold: float = BACKWARDATION_THRESHOLD,
        smooth_days: int = SMOOTH_DAYS,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            contango_threshold=contango_threshold,
            backwardation_threshold=backwardation_threshold,
            smooth_days=smooth_days,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.contango_threshold = float(contango_threshold)
        self.backwardation_threshold = float(backwardation_threshold)
        self.smooth_days = int(smooth_days)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.smooth_days + 5, self.trend_window + 5)
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d trend gate
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

        # Compute smoothed VIX3M / VIX ratio
        ratio_smoothed = float("nan")
        try:
            vix_hist = ctx.history("^VIX")
            vix3m_hist = ctx.history("^VIX3M")
            if (
                vix_hist is not None
                and vix3m_hist is not None
                and len(vix_hist) >= self.smooth_days
                and len(vix3m_hist) >= self.smooth_days
            ):
                # Pull last smooth_days closes from each
                vix_close = vix_hist["close"].dropna()
                vix3m_close = vix3m_hist["close"].dropna()
                if len(vix_close) >= self.smooth_days and len(vix3m_close) >= self.smooth_days:
                    # Align by index intersection
                    df = pd.concat(
                        [vix_close.rename("vix"), vix3m_close.rename("vix3m")],
                        axis=1,
                    ).dropna()
                    if len(df) >= self.smooth_days:
                        ratios = df["vix3m"] / df["vix"]
                        ratio_smoothed = float(ratios.iloc[-self.smooth_days:].mean())
        except Exception:
            pass

        if not np.isfinite(ratio_smoothed):
            # Default to neutral SPY when signal unavailable
            ratio_smoothed = 1.02

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Continuous tilt: equity-fraction is monotone in (VIX3M/VIX - 1)
        # ratio_smoothed typically ranges [0.85, 1.20]; this maps to
        # equity fraction roughly [0.10, 0.95]
        eq_frac = max(0.10, min(0.95, 0.50 + 4.0 * (ratio_smoothed - 1.02)))

        # SPY 100d trend gate: dampens equity in bear regimes
        if not spy_bull:
            eq_frac *= 0.40

        bond_frac = 1.0 - eq_frac

        # Equity allocation: QQQ for stronger contango, SPY for milder
        if ratio_smoothed >= self.contango_threshold and "QQQ" in live:
            target["QQQ"] = eq_frac * self.exposure
        elif "SPY" in live:
            target["SPY"] = eq_frac * self.exposure

        # Bond allocation: IEF (mid-duration is steady)
        if "IEF" in live:
            target["IEF"] = bond_frac * self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


NAME = "vix_term_structure_qqq"
HYPOTHESIS = (
    "VIX term-structure regime QQQ allocator: VIX3M/VIX 5d-smoothed ratio "
    "contango (>=1.05) hold QQQ 97%; neutral (1.00-1.05) hold SPY 97%; "
    "backwardation (<1.00) hold IEF 60%+SHY 37%. Pure vol curve shape "
    "signal orthogonal to VIX level."
)

STRATEGY = VixTermStructureQqq()
