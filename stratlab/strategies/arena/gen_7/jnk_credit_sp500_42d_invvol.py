"""JNK 50d SMA Credit Gate + SP500 42d Momentum Inverse-Vol — gen_7 sonnet-3

Hypothesis: Hold top-15 SP500 stocks by 42d return (inverse-vol weighted)
when JNK above 50d SMA AND SPY above 200d SMA (dual credit+trend gate);
rotate to TLT 60%+GLD 37% when either signal is negative; monthly rebalance.

Rationale: JNK 50d SMA is a clean, smoothed credit signal — when HY bonds
are trending up, credit conditions are favorable for equity risk-on. Combining
with SPY 200d SMA gives a dual gate that avoids both credit blowups and equity
bear markets. The 42d momentum window is shorter than the established 126d
skip-month strategies, capturing more recent momentum with inverse-vol weighting
to avoid concentration in the highest-beta winners.

Distinction from existing: credit_gated_skipmon uses JNK/LQD 30d return ratio
and 126d-skip-21d momentum; this uses JNK 50d SMA (trend-based, not relative-to-LQD)
and 42d momentum without skip-month; TLT+GLD refuge (not just TLT).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21      # monthly
MOMENTUM_WINDOW = 42      # 42d return
VOL_WINDOW = 20           # for inverse-vol weights
TOP_K = 15
TREND_WINDOW = 200        # SPY 200d SMA
JNK_MA = 50               # JNK 50d SMA
EXPOSURE = 0.97
TLT_WEIGHT = 0.60
GLD_WEIGHT = 0.37


class JnkCreditSp500_42dInvvol(Strategy):
    """JNK 50d SMA gated SP500 42d momentum inverse-vol strategy."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        jnk_ma: int = JNK_MA,
        exposure: float = EXPOSURE,
        tlt_weight: float = TLT_WEIGHT,
        gld_weight: float = GLD_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            jnk_ma=jnk_ma,
            exposure=exposure,
            tlt_weight=tlt_weight,
            gld_weight=gld_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.jnk_ma = int(jnk_ma)
        self.exposure = float(exposure)
        self.tlt_weight = float(tlt_weight)
        self.gld_weight = float(gld_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.jnk_ma, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # JNK 50d SMA credit gate
        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return []
        if len(jnk_hist) < self.jnk_ma + 5:
            return []
        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) < self.jnk_ma:
            return []
        jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
        credit_ok = float(jnk_close.iloc[-1]) > jnk_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull or not credit_ok:
            # Defensive: TLT 60% + GLD 37%
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure * self.tlt_weight
            if "GLD" in closes_now.index:
                target["GLD"] = self.exposure * self.gld_weight
        else:
            # Risk-on: top-15 SP500 by 42d momentum, inverse-vol weighted
            need = max(self.momentum_window, self.vol_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # 42d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Not enough candidates — TLT+GLD defensive
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure * self.tlt_weight
                if "GLD" in closes_now.index:
                    target["GLD"] = self.exposure * self.gld_weight
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "GLD", "SPY", "JNK"]


NAME = "jnk_credit_sp500_42d_invvol"
HYPOTHESIS = (
    "JNK 50d SMA credit gate + SP500 42d momentum inverse-vol weighted: "
    "hold top-15 SP500 stocks by 42d return (inverse-vol weighted) when JNK above 50d SMA "
    "AND SPY above 200d SMA; rotate to TLT 60%+GLD 37% when either signal is negative; "
    "monthly rebalance"
)

UNIVERSE = _universe

STRATEGY = JnkCreditSp500_42dInvvol()
