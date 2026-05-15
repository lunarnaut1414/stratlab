"""SP500 momentum with below-median realized-vol quality filter.

Hypothesis (sonnet-1, gen_10):
    Hold top-15 SP500 stocks by 126d momentum where each stock must have
    21d realized vol BELOW the cross-sectional median vol of all SP500 stocks.
    This eliminates high-volatility momentum names (lottery stocks, event-driven
    spikes) that tend to mean-revert. It's a RELATIVE quality filter (vs universe)
    rather than an absolute threshold (like RSI > 40).

Rationale:
  - Pure momentum often selects extremely volatile stocks at the tail: stocks
    with high 6-month returns often got there via a volatile spike. Below-median
    vol screen removes these while keeping steady-growers.
  - The relative filter (vs universe median) is self-calibrating across regimes:
    in high-vol markets, the "allowed" vol ceiling rises proportionally, so the
    strategy doesn't sit fully in IEF during broad market stress.
  - Different from gen6_sp500_lowvol_factor (which SORTS by lowest vol, picks top-20
    lowest): this strategy uses momentum as primary ranking and vol as a filter gate.
  - Different from gen9_sp500_rsi_quality_momentum (RSI>35 absolute threshold).
  - Mechanism doesn't depend on calm-VIX regime: the relative comparison persists
    across all vol regimes. OOS retention expected: HIGH.

Design:
  1. Compute 21d realized vol for all SP500 stocks.
  2. Compute the cross-sectional median vol.
  3. Exclude stocks with vol > median.
  4. From remaining, rank by 126d momentum; hold top-15.
  5. Inverse-vol weighted. SPY 200d SMA gate to IEF. Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 126    # 6-month momentum
VOL_WINDOW = 21          # for realized vol filter AND inverse-vol weights
SPY_TREND_WINDOW = 200   # outer trend gate
TOP_K = 15
EXPOSURE = 0.97


class SP500SubmedvolMomentum(Strategy):
    """SP500 126d momentum with below-median 21d vol quality filter.

    Inverse-vol weighted. SPY 200d bear gate to IEF. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = self.momentum_window + self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 2:
                return []

            # Compute realized vol for ALL stocks first (for median)
            all_vols: dict[str, float] = {}
            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.vol_window + 1:
                    continue
                # Realized vol
                tail = col.values[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                all_vols[sym] = rv

                # 126d momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue
                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(all_vols) < 20:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                # Compute cross-sectional median vol
                median_vol = float(np.median(list(all_vols.values())))

                # Filter: only stocks with vol <= median (below-median vol)
                qualified = {
                    sym: ret
                    for sym, ret in scores.items()
                    if sym in all_vols and all_vols[sym] <= median_vol
                }

                if len(qualified) < 5:
                    if "IEF" in closes_now.index:
                        target["IEF"] = self.exposure
                else:
                    k = min(self.top_k, len(qualified))
                    ranked = sorted(qualified, key=qualified.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        return []
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # --- Build orders ---
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_submedvol_momentum"
HYPOTHESIS = (
    "SP500 top-15 momentum with below-median-vol quality filter: rank by 126d momentum, "
    "hold only stocks with 21d realized vol below the universe median vol (eliminates high-vol "
    "momentum names prone to reversal); inverse-vol weighted; SPY 200d gate to IEF; biweekly "
    "rebalance — relative-vol quality filter unlike absolute RSI or SMA screens"
)

UNIVERSE = _universe

STRATEGY = SP500SubmedvolMomentum()
