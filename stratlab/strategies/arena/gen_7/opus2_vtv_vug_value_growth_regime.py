"""VTV/VUG value-vs-growth regime rotation — gen_7 opus-2 (gap_finder).

Hypothesis: The relative strength between Vanguard Value (VTV) and Vanguard
Growth (VUG) ETFs identifies which factor regime is leading. Hold the
factor-aligned vehicle: when growth leads, QQQ (concentrated growth/tech);
when value leads with bull market, SPY+VTV value tilt; bear regime defaults
to TLT.

Logic:
  - Compute VUG and VTV 60d returns.
  - VUG_ret > VTV_ret AND SPY > 200d -> QQQ 97% (growth-leadership regime).
  - VTV_ret > VUG_ret AND SPY > 200d -> SPY 60% + VTV 37% (value tilt).
  - SPY < 200d -> TLT 97% (bear regime).

Distinction: existing strategies have factor-ETF rotation (failed) and
sector-rotation (saturated). None use the binary VUG-VTV regime as a switch
between QQQ and value-tilted SPY. VTV/VUG inception 2004; both have full
2010-2018 coverage. Pure ETF-vehicle strategy (no SP500 stock ranking) so
risk profile is structurally different from leaderboard SP500-xsect cluster.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5  # weekly
RATIO_WINDOW = 60
TREND_WINDOW = 200
EXPOSURE = 0.97


class VtvVugValueGrowthRegime(Strategy):
    """VTV vs VUG 60d return regime rotates QQQ / SPY+VTV / TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ratio_window: int = RATIO_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ratio_window=ratio_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.ratio_window = int(ratio_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.ratio_window) + 10
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

        # SPY 200d gate
        bull_market = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    bull_market = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # VUG vs VTV 60d return regime
        growth_leads = True
        signal_ok = False
        try:
            vug_hist = ctx.history("VUG")
            vtv_hist = ctx.history("VTV")
            if (vug_hist is not None and vtv_hist is not None
                    and len(vug_hist) >= self.ratio_window
                    and len(vtv_hist) >= self.ratio_window):
                vug_close = vug_hist["close"].dropna()
                vtv_close = vtv_hist["close"].dropna()
                if (len(vug_close) >= self.ratio_window
                        and len(vtv_close) >= self.ratio_window):
                    vug_ret = float(vug_close.iloc[-1] / vug_close.iloc[-self.ratio_window] - 1.0)
                    vtv_ret = float(vtv_close.iloc[-1] / vtv_close.iloc[-self.ratio_window] - 1.0)
                    growth_leads = (vug_ret > vtv_ret)
                    signal_ok = True
        except Exception:
            pass

        target: dict[str, float] = {}
        if not bull_market:
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif signal_ok and growth_leads:
            # Growth-leadership: QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # Value-leadership (or signal unavailable): SPY 60% + VTV 37%
            for sym, w in [("SPY", 0.60), ("VTV", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure

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


NAME = "opus2_vtv_vug_value_growth_regime"
HYPOTHESIS = (
    "VTV vs VUG 60d-return regime: VUG > VTV AND SPY > 200d hold QQQ 97%; VTV > VUG with bull "
    "hold SPY 60%+VTV 37% (value tilt); SPY < 200d hold TLT; weekly rebalance; binary value/growth switch."
)
UNIVERSE = ["QQQ", "SPY", "VTV", "VUG", "TLT"]

STRATEGY = VtvVugValueGrowthRegime()
