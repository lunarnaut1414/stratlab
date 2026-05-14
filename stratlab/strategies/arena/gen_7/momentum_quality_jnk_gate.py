"""Momentum-quality composite SP500 with JNK credit gate.

Hypothesis: Score SP500 stocks by combining 63d momentum with a price-stability
factor (inverse of normalized 20d vol). High-scoring stocks are consistent
momentum names without excessive noise. Hold top-15 when credit conditions are
favorable (JNK above its 20d SMA) AND equity trend is up (SPY above 200d SMA).
Rotate to IEF+GLD when credit conditions deteriorate.

Rationale: Pure momentum picks high-vol lottery names. Weighting by price
stability (low vol relative to 20d range) filters toward quality momentum —
stocks making steady, persistent gains rather than spiking. The JNK gate
avoids holding equities when credit markets signal stress, which is orthogonal
to VIX timing.

Distinction from existing strategies:
  - Uses momentum * inverse_vol composite score (not just raw momentum)
  - JNK 20d SMA credit gate (existing gen6_jnk_vix_dual_gate_qqq uses 20d SMA
    but with VIX dual gate and routes to QQQ, not individual stocks)
  - IEF+GLD defensive (not TLT or SHY — different duration/asset mix)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21    # monthly
MOM_WINDOW = 63         # 3-month momentum
VOL_WINDOW = 20         # 20d vol for stability factor
JNK_MA = 20             # JNK credit gate
TREND_WINDOW = 200      # SPY 200d SMA
TOP_K = 15
EXPOSURE = 0.97


class MomentumQualityJnkGate(Strategy):
    """Momentum * price-stability composite with JNK credit gate; IEF+GLD defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        vol_window: int = VOL_WINDOW,
        jnk_ma: int = JNK_MA,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            vol_window=vol_window,
            jnk_ma=jnk_ma,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.vol_window = int(vol_window)
        self.jnk_ma = int(jnk_ma)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 10
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

        # SPY 200d SMA gate
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

        # JNK credit gate
        credit_ok = True
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    credit_ok = float(jnk_close.iloc[-1]) > jnk_ma_val
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull_market or not credit_ok:
            # Defensive: IEF 60% + GLD 37%
            for sym, w in [("IEF", 0.60), ("GLD", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: momentum * stability composite
            need = self.mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_window:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window + self.vol_window:
                        continue

                    # 63d momentum
                    ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                    if not np.isfinite(ret):
                        continue

                    # Price stability: 1 / (20d vol * sqrt(252))
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv_ann = float(np.std(logr) * np.sqrt(252))
                    if rv_ann <= 1e-6 or not np.isfinite(rv_ann):
                        continue
                    stability = 1.0 / rv_ann

                    # Only consider positive momentum stocks
                    if ret > 0:
                        scores[sym] = ret * stability

                if len(scores) < 5:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:min(self.top_k, len(ranked))]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
                        target[sym] = per_w

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
    return sp500_tickers() + ["JNK", "IEF", "GLD", "SPY"]


NAME = "momentum_quality_jnk_gate"
HYPOTHESIS = (
    "Momentum-quality composite SP500 with JNK credit gate: score stocks by 63d momentum "
    "times price-stability (1 minus 20d normalized vol); hold top-15 when JNK above 20d SMA "
    "AND SPY above 200d SMA; rotate to IEF+GLD 60/37 otherwise; monthly rebalance"
)

UNIVERSE = _universe

STRATEGY = MomentumQualityJnkGate()
