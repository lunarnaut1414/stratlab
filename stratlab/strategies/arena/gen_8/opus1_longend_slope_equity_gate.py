"""opus-1 / gen_8 — Long-end Yield Slope Equity Gate

Mutation of gen8_tnx_yield_trend_equity_gate (IS Calmar 1.01, h1/h2 0.98/1.45).

Parent uses TNX (10Y yield) vs its own 200d MA as the equity regime gate.
This variant changes the *macro signal* but keeps the same gating logic:
use the LONG-END slope (^TYX - ^TNX, i.e. 30Y minus 10Y) vs its own 200d MA.

Rationale: the 30Y-10Y term premium captures duration-risk-premium dynamics
that TNX alone misses. When the long end is steep (term premium expanding,
TYX-TNX > its 200d MA), bond markets are pricing in higher long-run growth
and inflation — a risk-on signal for equity. When the long end flattens
(TYX-TNX < its 200d MA), the bond market is signalling lower long-run growth
expectations — rotate to SPY+TLT blend.

Same SPY 200d outer bear gate, same top-15 SP500 momentum, same biweekly
rebalance. The only change is the macro signal itself.

Coverage check:
- ^TYX available from 1977 (deep coverage, no warmup issue for IS 2010-2018)
- ^TNX available from 1962
- 200d MA on the slope means signal becomes live ~200 trading days into IS
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
SLOPE_TREND_WINDOW = 200
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_TYX = "^TYX"
_TNX = "^TNX"


class LongEndSlopeEquityGate(Strategy):
    """SP500 momentum gated by 30Y-10Y term premium vs its 200d MA."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        slope_trend_window: int = SLOPE_TREND_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            slope_trend_window=slope_trend_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.slope_trend_window = int(slope_trend_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slope_trend_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- 30Y-10Y long-end slope vs own 200d MA ---
        slope_steep = True  # default risk-on if signal unavailable
        try:
            tyx_hist = ctx.history(_TYX)
            tnx_hist = ctx.history(_TNX)
            if (tyx_hist is not None and tnx_hist is not None and
                    len(tyx_hist) >= self.slope_trend_window + 2 and
                    len(tnx_hist) >= self.slope_trend_window + 2):
                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()
                # align by tail
                n = min(len(tyx_close), len(tnx_close))
                if n >= self.slope_trend_window + 1:
                    slope = tyx_close.values[-n:] - tnx_close.values[-n:]
                    slope_ma = float(np.mean(slope[-self.slope_trend_window:]))
                    slope_now = float(slope[-1])
                    slope_steep = slope_now > slope_ma
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # SPY in bear — fully defensive TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not slope_steep:
            # Flat/inverted long-end slope: SPY+TLT blend (mirror of parent's
            # rising-rates fallback — equity exposure but de-concentrated)
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _TLT in live:
                target[_TLT] = self.exposure * 0.382
        else:
            # Steep long-end slope + SPY bull — top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY, _TYX, _TNX]


NAME = "opus1_longend_slope_equity_gate"
HYPOTHESIS = (
    "Macro-signal mutation of tnx_yield_trend_equity_gate: replace TNX 200d MA gate with "
    "30Y-10Y long-end slope (TYX-TNX) vs its own 200d MA gating SP500 top-15 momentum vs "
    "SPY 60%+TLT 38% blend; SPY 200d outer bear gate to TLT; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = LongEndSlopeEquityGate()
