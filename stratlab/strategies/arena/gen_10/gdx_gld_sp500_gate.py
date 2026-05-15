"""GDX-vs-IAU commodity signal gating SP500 momentum — gen_10 sonnet-7

Hypothesis: GDX (gold miners ETF) outperforming IAU (physical gold) over a
42-day window signals risk-on real-economy expansion. When miners lead gold
AND SPY is in bull trend: hold top-15 SP500 stocks by 63d momentum (inverse-
vol weighted). When IAU leads GDX (gold physical demand > miners = risk-off):
hold XLU (defensive utilities) 50% + TLT 47%. SPY 200d outer bear gate
always goes to TLT.

Rationale:
  - GDX/IAU signal is orthogonal to yield-curve signals, VIX-level gates,
    credit-spread gates, and EM/US flow gates already on leaderboard.
  - When GDX leads IAU: economic activity is healthy, risk-appetite exists,
    stock selection momentum should work.
  - When IAU leads GDX: flight to physical gold, risk-off, hold defensive.
  - The defensive branch (XLU+TLT) creates a different return path than
    SPY+IEF or TLT-only defensives used by other strategies.
  - This is NOT a macro-signal allocator in the degraded-OOS sense: the
    GDX/IAU ratio is a relative-value commodity signal rather than an
    absolute-macro indicator (like VIX level or yield curve slope).
  - Expected lower corr to top-5 because: (a) defensive branch is XLU+TLT
    not SPY+IEF, and (b) 42d GDX/IAU signal fires ~45-55% of time, so
    roughly half the IS days are in non-stock mode.

Data checks:
  - GDX: starts 2006-05-22, covers IS (2010-01-01 onwards). OK.
  - IAU: starts 2005-01-28, covers IS. OK.
  - XLU: starts 1998, covers IS. OK.
  - TLT: starts 2002, covers IS. OK.
  - SPY: covers IS. OK.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
SIGNAL_WINDOW = 42        # GDX vs IAU comparison window
MOMENTUM_WINDOW = 63      # stock momentum window when risk-on
VOL_WINDOW = 21           # inverse-vol for stock sizing
SPY_TREND_WINDOW = 200    # outer bear gate
TOP_K = 15
EXPOSURE = 0.97
DEFENSIVE_XLU_W = 0.50   # XLU weight in risk-off regime
DEFENSIVE_TLT_W = 0.47   # TLT weight in risk-off regime


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["GDX", "IAU", "XLU", "TLT", "SPY"]


UNIVERSE = _universe


class GdxGldSP500Gate(Strategy):
    """GDX vs IAU 42d return gate: risk-on -> SP500 63d momentum top-15;
    risk-off -> XLU+TLT; SPY 200d bear gate to TLT; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        signal_window: int = SIGNAL_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        defensive_xlu_w: float = DEFENSIVE_XLU_W,
        defensive_tlt_w: float = DEFENSIVE_TLT_W,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            signal_window=signal_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            defensive_xlu_w=defensive_xlu_w,
            defensive_tlt_w=defensive_tlt_w,
        )
        self.rebalance_every = int(rebalance_every)
        self.signal_window = int(signal_window)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.defensive_xlu_w = float(defensive_xlu_w)
        self.defensive_tlt_w = float(defensive_tlt_w)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.signal_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d outer bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # GDX vs IAU 42d return signal
            gdx_hist = ctx.history("GDX")
            iau_hist = ctx.history("IAU")

            risk_on = False  # default to risk-off if data missing
            if len(gdx_hist) >= self.signal_window + 2 and len(iau_hist) >= self.signal_window + 2:
                gdx_close = gdx_hist["close"].dropna()
                iau_close = iau_hist["close"].dropna()
                if (len(gdx_close) >= self.signal_window + 1 and
                        len(iau_close) >= self.signal_window + 1):
                    gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-self.signal_window] - 1.0)
                    iau_ret = float(iau_close.iloc[-1] / iau_close.iloc[-self.signal_window] - 1.0)
                    if np.isfinite(gdx_ret) and np.isfinite(iau_ret):
                        risk_on = gdx_ret > iau_ret

            if not risk_on:
                # Risk-off: XLU + TLT defensive blend
                if "XLU" in live:
                    target["XLU"] = self.defensive_xlu_w
                if "TLT" in live:
                    target["TLT"] = self.defensive_tlt_w
                # Fallback if one missing
                if "XLU" not in live and "TLT" in live:
                    target["TLT"] = self.exposure
                elif "TLT" not in live and "XLU" in live:
                    target["XLU"] = self.exposure
            else:
                # Risk-on: SP500 momentum stock selection
                need = self.momentum_window + self.vol_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 5:
                    # Fallback to defensive
                    if "TLT" in live:
                        target["TLT"] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}

                    for sym in prices.columns:
                        if sym in ("GDX", "IAU", "XLU", "TLT", "SPY"):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + self.vol_window + 2:
                            continue

                        # 63d momentum
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-self.momentum_window])
                        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                            continue
                        ret = p_end / p_start - 1.0
                        if not np.isfinite(ret):
                            continue

                        # Inverse-vol weight
                        tail = col.values[-(self.vol_window + 1):]
                        if len(tail) < self.vol_window + 1:
                            continue
                        logr = np.log(tail[1:] / tail[:-1])
                        rv = float(np.std(logr))
                        if rv <= 1e-6 or not np.isfinite(rv):
                            continue

                        scores[sym] = ret
                        inv_vols[sym] = 1.0 / rv

                    if len(scores) < 5:
                        if "TLT" in live:
                            target["TLT"] = self.exposure
                    else:
                        k = min(self.top_k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                        iv_sum = sum(inv_vols[s] for s in ranked)
                        if iv_sum <= 0:
                            return []
                        for sym in ranked:
                            target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


NAME = "gdx_gld_sp500_gate"
HYPOTHESIS = (
    "GDX-vs-IAU 42d return gate for SP500 momentum: when GDX outperforms IAU (risk-on miners "
    "leading physical gold), hold top-15 SP500 stocks by 63d momentum inverse-vol weighted; "
    "when IAU leads GDX (flight to physical gold, risk-off), hold XLU 50%+TLT 47%; SPY 200d "
    "outer bear gate to TLT; biweekly rebalance — commodity-market signal gates stock selection"
)

STRATEGY = GdxGldSP500Gate()
