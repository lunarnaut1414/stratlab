"""GLD-SPY 30d rolling correlation regime — gen_7 sonnet-8

Hypothesis: The correlation between gold (GLD) and SPY typically is near zero or
negative during normal market conditions (gold as diversifier). During crisis periods,
both assets can fall together (positive correlation) as forced liquidations occur —
this "crisis correlation" regime signals genuine systemic risk.

Signal: 30d rolling return correlation of GLD and SPY daily returns.
  - Negative correlation (< -0.05): normal regime → hold top-15 SP500 momentum stocks
  - Mildly positive (−0.05 to +0.25): transitional → hold SPY 97%
  - Strongly positive (> +0.25): crisis correlation → rotate to SHY 97%

Additional gate: when SPY is below its 200d SMA (bear market), regardless of
correlation regime, hold TLT 97%.

Biweekly rebalance. The cross-asset correlation signal is novel vs existing
VIX-level, credit-spread, breadth, and yield-curve regime signals on the leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly
CORR_WINDOW = 30         # 30d rolling correlation
MOMENTUM_WINDOW = 63     # 63d momentum for stock selection
TREND_WINDOW = 200       # SPY 200d SMA gate
TOP_K = 15
EXPOSURE = 0.97
CORR_CRISIS = 0.25       # positive corr threshold = crisis
CORR_NORMAL = -0.05      # negative corr threshold = normal


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["GLD", "SPY", "TLT", "SHY"]


class GldSpyCorrRegime(Strategy):
    """GLD-SPY rolling correlation regime: momentum stocks / SPY / SHY based on GLD-SPY corr."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        corr_window: int = CORR_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        corr_crisis: float = CORR_CRISIS,
        corr_normal: float = CORR_NORMAL,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            corr_window=corr_window,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            corr_crisis=corr_crisis,
            corr_normal=corr_normal,
        )
        self.rebalance_every = int(rebalance_every)
        self.corr_window = int(corr_window)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.corr_crisis = float(corr_crisis)
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

        # SPY 200d SMA gate (bear market → TLT)
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
            # Bear market: hold TLT
            target: dict[str, float] = {}
            if "TLT" in live:
                target["TLT"] = self.exposure
            return self._build_orders(ctx, target, live, equity)

        # Compute 30d rolling correlation of GLD and SPY daily returns
        corr_value = float("nan")
        try:
            gld_hist = ctx.history("GLD")
            spy_hist2 = ctx.history("SPY")
            need = self.corr_window + 5
            if len(gld_hist) >= need and len(spy_hist2) >= need:
                gld_close = gld_hist["close"].dropna().values
                spy_close2 = spy_hist2["close"].dropna().values
                n = min(len(gld_close), len(spy_close2), need)
                gld_arr = gld_close[-n:]
                spy_arr = spy_close2[-n:]
                if len(gld_arr) >= self.corr_window + 1:
                    gld_ret = np.diff(np.log(gld_arr[-self.corr_window - 1:]))
                    spy_ret = np.diff(np.log(spy_arr[-self.corr_window - 1:]))
                    if len(gld_ret) >= 10 and len(spy_ret) >= 10:
                        # Compute Pearson correlation
                        corr_val = float(np.corrcoef(gld_ret, spy_ret)[0, 1])
                        if np.isfinite(corr_val):
                            corr_value = corr_val
        except Exception:
            pass

        target = {}

        if np.isnan(corr_value):
            # Signal unavailable: default to SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
        elif corr_value > self.corr_crisis:
            # Crisis correlation: both assets falling together → SHY (cash-like)
            if "SHY" in live:
                target["SHY"] = self.exposure
        elif corr_value < self.corr_normal:
            # Normal regime (negative corr, gold as diversifier): momentum stocks
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                skip_set = {"GLD", "SPY", "TLT", "SHY", "QQQ", "IEF", "IWM",
                            "TIP", "AGG", "BIL", "SHV", "LQD", "HYG", "JNK",
                            "VNQ", "EEM", "EFA", "RSP", "MDY", "IJH", "IJR",
                            "SSO", "TQQQ", "UPRO", "VUG", "VLUE", "VTV", "MTUM",
                            "USMV", "IVE", "IWN", "IWP", "XLK", "XLV", "XLF",
                            "XLI", "XLP", "XLU", "XLE", "XLB", "XLY", "XLRE", "XLC",
                            "DBC", "IAU", "SLV", "USO", "GDX"}
                for sym in prices.columns:
                    if sym in skip_set:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window + 1:
                        continue
                    p_end = float(col.values[-1])
                    p_start = float(col.values[-self.momentum_window])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    if "SPY" in live:
                        target["SPY"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight
        else:
            # Transitional regime: hold SPY
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


NAME = "gld_spy_corr_regime"
HYPOTHESIS = (
    "GLD-SPY 30d rolling correlation regime: when corr is negative (flight-to-safety, normal regime) "
    "hold top-15 SP500 stocks by 63d momentum; when corr is positive (crisis correlation, both assets "
    "falling together) rotate to SHY 97%; SPY 200d SMA bear gate to TLT; biweekly rebalance; "
    "cross-asset correlation as novel signal"
)

UNIVERSE = _universe

STRATEGY = GldSpyCorrRegime()
