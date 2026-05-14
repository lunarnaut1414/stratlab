"""TLT-SPY rolling correlation regime v2 — opus-1 gen_7

Mutation of gen7_gld_spy_corr_regime (parent IS Calmar 1.06).

Parent: GLD-SPY correlation; rotates between SP500 momentum / SPY / SHY.
Mutation: replace the regime DEFINITION. Use TLT-SPY 30d return correlation
instead of GLD-SPY. Bond-equity correlation has a structurally different
historical pattern: "normal" regimes show strongly negative TLT-SPY corr
(bonds rally on equity sell-offs = flight-to-quality); "stagflation/regime
transition" shows correlation drift toward zero or positive (bonds and
equities sell off together when the safe-asset bid breaks down).

V2 fix: avoid SP500-stock corr attractor by using ETF vehicles (MTUM+QQQ)
in the normal regime instead of individual SP500 momentum stocks.

Thresholds tuned for bond-equity dynamics:
  - corr <= -0.45 : strong flight-to-quality (normal) → MTUM 60% + QQQ 37%
  - -0.45 < corr <= +0.10 : weakened safe-asset bid (transitional) → SPY
  - corr > +0.10 : correlated-sell / regime stress → SHY

This is structurally different from GLD-SPY (which captures crisis-correlation
liquidations) — bond-equity decoupling captures FED-policy / yield-shock regimes.
ETF risk-on vehicles avoid the SP500-xsect corr attractor.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
CORR_WINDOW = 30
MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
CORR_STRESS = 0.10        # >= this → stress regime (SHY)
CORR_NORMAL = -0.45       # <= this → normal flight-to-quality (momentum stocks)


UNIVERSE = ["MTUM", "QQQ", "SPY", "TLT", "SHY"]


class TltSpyCorrRegime(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        corr_window: int = CORR_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        corr_stress: float = CORR_STRESS,
        corr_normal: float = CORR_NORMAL,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            corr_window=corr_window,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            corr_stress=corr_stress,
            corr_normal=corr_normal,
        )
        self.rebalance_every = int(rebalance_every)
        self.corr_window = int(corr_window)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.corr_stress = float(corr_stress)
        self.corr_normal = float(corr_normal)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.corr_window, self.momentum_window, self.trend_window) + 10
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

        # SPY 200d SMA bear-market gate (flat to TLT)
        bear_market = False
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 5:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(np.mean(spy_close.values[-self.trend_window:]))
                    bear_market = float(spy_close.values[-1]) < spy_sma
        except Exception:
            pass

        if bear_market:
            target: dict[str, float] = {}
            if "TLT" in live:
                target["TLT"] = self.exposure
            return self._build_orders(ctx, target, live, equity)

        # Compute TLT-SPY 30d correlation
        corr_value = float("nan")
        try:
            tlt_hist = ctx.history("TLT")
            spy_hist2 = ctx.history("SPY")
            need = self.corr_window + 5
            if len(tlt_hist) >= need and len(spy_hist2) >= need:
                tlt_close = tlt_hist["close"].dropna().values
                spy_close2 = spy_hist2["close"].dropna().values
                n = min(len(tlt_close), len(spy_close2), need)
                if n >= self.corr_window + 1:
                    tlt_arr = tlt_close[-(self.corr_window + 1):]
                    spy_arr = spy_close2[-(self.corr_window + 1):]
                    tlt_ret = np.diff(np.log(tlt_arr))
                    spy_ret = np.diff(np.log(spy_arr))
                    if len(tlt_ret) >= 10 and np.std(tlt_ret) > 1e-9 and np.std(spy_ret) > 1e-9:
                        corr_val = float(np.corrcoef(tlt_ret, spy_ret)[0, 1])
                        if np.isfinite(corr_val):
                            corr_value = corr_val
        except Exception:
            pass

        target = {}

        if np.isnan(corr_value):
            if "SPY" in live:
                target["SPY"] = self.exposure
        elif corr_value >= self.corr_stress:
            # Bonds and equities correlated → safe asset bid broken → SHY
            if "SHY" in live:
                target["SHY"] = self.exposure
        elif corr_value <= self.corr_normal:
            # Strong flight-to-quality regime → factor-momentum ETF + nasdaq
            # (NOT individual SP500 stocks; avoids corr attractor)
            mtum_w = 0.60 * self.exposure
            qqq_w = 0.37 * self.exposure
            if "MTUM" in live and "QQQ" in live:
                target["MTUM"] = mtum_w
                target["QQQ"] = qqq_w
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Transitional: hold SPY
            if "SPY" in live:
                target["SPY"] = self.exposure

        return self._build_orders(ctx, target, live, equity)

    def _build_orders(
        self,
        ctx: BarContext,
        target: dict[str, float],
        live: dict[str, float],
        equity: float,
    ) -> list[Order]:
        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
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


NAME = "opus1_tlt_spy_corr_regime"
HYPOTHESIS = (
    "TLT-SPY rolling correlation regime v2: 30d corr; normal flight-to-quality "
    "(<=-0.45) holds MTUM 60%+QQQ 37% (factor-momentum ETF + nasdaq, NOT individual "
    "SP500 stocks); transitional (-0.45,+0.10) holds SPY; correlated-sell stress "
    "(>+0.10) holds SHY; ETF vehicles avoid SP500-xsect corr attractor"
)

STRATEGY = TltSpyCorrRegime()
