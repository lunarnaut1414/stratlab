"""SMH vs KBE industry-leadership rotation — gen_7 opus-2 (gap_finder).

Hypothesis: Sub-industry ETFs SMH (semiconductors) and KBE (banks) capture
two distinct phases of the cycle: semis lead in expansion/secular-growth
phase, banks lead late-cycle when curve steepens. Their 63d return spread
is a sub-industry leadership signal not yet on the leaderboard.

Logic:
  - 63d return of SMH and KBE.
  - SMH leads (SMH_ret > KBE_ret) AND SPY > 200d -> QQQ 97% (secular tech).
  - KBE leads AND SMH_ret > 0 AND SPY > 200d -> SPY 60% + XLF 37% (financials tilt).
  - SMH < 0 AND KBE < 0 (both negative) OR SPY < 200d -> TLT 60% + SHY 37%.
  - Biweekly rebalance.

Distinction: sub-industry pair-leadership signal. Existing leaderboard has
no SMH or KBE based strategies. SMH (2000), KBE (2005), XLF (2000) all have
full IS coverage.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
RATIO_WINDOW = 63
TREND_WINDOW = 200
EXPOSURE = 0.97


class SmhKbeIndustryLeadership(Strategy):
    """SMH vs KBE 63d return leadership rotates QQQ / SPY+XLF / defensive."""

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

        # SMH and KBE 63d returns
        smh_ret = float("nan")
        kbe_ret = float("nan")
        try:
            smh_hist = ctx.history("SMH")
            if smh_hist is not None and len(smh_hist) >= self.ratio_window:
                c = smh_hist["close"].dropna()
                if len(c) >= self.ratio_window:
                    smh_ret = float(c.iloc[-1] / c.iloc[-self.ratio_window] - 1.0)
        except Exception:
            pass
        try:
            kbe_hist = ctx.history("KBE")
            if kbe_hist is not None and len(kbe_hist) >= self.ratio_window:
                c = kbe_hist["close"].dropna()
                if len(c) >= self.ratio_window:
                    kbe_ret = float(c.iloc[-1] / c.iloc[-self.ratio_window] - 1.0)
        except Exception:
            pass

        target: dict[str, float] = {}
        # Decision tree
        if not bull_market:
            for sym, w in [("TLT", 0.60), ("SHY", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        elif np.isnan(smh_ret) or np.isnan(kbe_ret):
            # Signal unavailable: default SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        elif smh_ret < 0 and kbe_ret < 0:
            # Both negative: defensive
            for sym, w in [("TLT", 0.60), ("SHY", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        elif smh_ret > kbe_ret:
            # Semis lead -> QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # Banks lead -> SPY + XLF
            for sym, w in [("SPY", 0.60), ("XLF", 0.37)]:
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


NAME = "opus2_smh_kbe_industry_leadership"
HYPOTHESIS = (
    "SMH vs KBE 63d return leadership: SMH leads with bull -> QQQ 97%; KBE leads with both positive "
    "and bull -> SPY 60%+XLF 37%; both negative or bear -> TLT 60%+SHY 37%; biweekly rebalance."
)
UNIVERSE = ["QQQ", "SPY", "XLF", "SMH", "KBE", "TLT", "SHY"]

STRATEGY = SmhKbeIndustryLeadership()
