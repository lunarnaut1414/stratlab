"""5Y-2Y Belly Curve Spread Gating SP500 Momentum — gen_8 opus-2 (gap_finder)

Hypothesis: Use the 5-year-minus-3-month Treasury spread (FVX - IRX, where
^FVX is the 5Y yield and ^IRX is the 13-week T-bill yield) as a "belly of
the curve" signal. This is the mid-curve segment where dealer roll-down
and carry trades operate, distinct from:

  - 10Y-2Y short-vs-mid spread (used by gen_7 yield_curve_slope_rotation
    and gen_8 rp_yield_curve_tilt)
  - 30Y-10Y long-end slope (used by gen_7 opus1_longend_curve_mtum_rotation)
  - 10Y-3M recession indicator (variant of 10Y-2Y)

The 5Y point is where rate-cycle inflection often appears first: the 2Y
anchors to Fed expectations, the 10Y reflects long-term growth, but the
5Y belly moves with intermediate growth and carry positioning. Using
FVX-IRX (5Y over 3M bills) captures intermediate vs short, providing a
medium-horizon carry-roll signal.

Three regimes:
  - Belly steep (FVX-IRX > 1.0%) AND SPY bull   : top-15 SP500 mom (97%)
       Carry-supported growth — equities rewarded
  - Belly flat (0 to 1%) AND SPY bull           : SPY 97%
       No edge — track the market
  - Belly inverted (<0%) OR SPY bear            : TLT 60% + IEF 37%
       Recession setup or already-bearish — defensive bonds

Why this fills a gap (after 4 rounds):
- 5Y-2Y belly is the ONE curve segment explicitly called out as untouched
  in the gen_8 brief's open-frontiers list.
- ^FVX cache from 1962 — full IS coverage. ^IRX cache start similar (long
  history). Both are signal-only series read via ctx.history().
- Routes between SP500 momentum (high IS Calmar but high corr) and ETF
  blends, balancing accept-gate constraints.

Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
SPY_TREND_WINDOW = 200
SMOOTH_DAYS = 20
STEEP_THRESHOLD = 1.0
FLAT_THRESHOLD = 0.0
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_FVX = "^FVX"  # 5Y yield (signal-only)
_IRX = "^IRX"  # 13-week T-bill yield (signal-only)


class BellyCurveFVX(Strategy):
    """SP500 momentum gated by 5Y-3M (FVX-IRX) belly curve spread."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        smooth_days: int = SMOOTH_DAYS,
        steep_threshold: float = STEEP_THRESHOLD,
        flat_threshold: float = FLAT_THRESHOLD,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            smooth_days=smooth_days,
            steep_threshold=steep_threshold,
            flat_threshold=flat_threshold,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.smooth_days = int(smooth_days)
        self.steep_threshold = float(steep_threshold)
        self.flat_threshold = float(flat_threshold)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.smooth_days, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_now = float(spy_close.iloc[-1])
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = spy_now > spy_sma

        # Compute 5Y-3M belly spread (smoothed)
        belly_spread = 1.0  # default neutral
        try:
            fvx_hist = ctx.history(_FVX)
            irx_hist = ctx.history(_IRX)
            if (fvx_hist is not None and len(fvx_hist) >= self.smooth_days + 2 and
                    irx_hist is not None and len(irx_hist) >= self.smooth_days + 2):
                fvx_close = fvx_hist["close"].dropna()
                irx_close = irx_hist["close"].dropna()
                if (len(fvx_close) >= self.smooth_days and
                        len(irx_close) >= self.smooth_days):
                    fvx_smooth = float(fvx_close.iloc[-self.smooth_days:].mean())
                    irx_smooth = float(irx_close.iloc[-self.smooth_days:].mean())
                    belly_spread = fvx_smooth - irx_smooth
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull or belly_spread < self.flat_threshold:
            # Bear or inverted belly — defensive
            if _TLT in live:
                target[_TLT] = self.exposure * 0.62
            if _IEF in live:
                target[_IEF] = self.exposure * 0.38
        elif belly_spread > self.steep_threshold:
            # Steep belly + SPY bull — top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    r = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(r):
                        scores[sym] = r
                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_w = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_w
        else:
            # Flat belly but SPY bull — SPY
            if _SPY in live:
                target[_SPY] = self.exposure

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
    return sp500_tickers() + [_SPY, _TLT, _IEF, _FVX, _IRX]


NAME = "opus2_belly_curve_5y2y_sp500"
HYPOTHESIS = (
    "5Y-3M Treasury belly spread (FVX-IRX) regime gating SP500 momentum: steep belly "
    "(>1%) AND SPY bull hold top-15 SP500 stocks by 63d momentum (carry-supported "
    "growth); flat belly (0-1%) AND SPY bull hold SPY 97%; inverted belly (<0%) OR "
    "SPY bear hold TLT 60%+IEF 37%. Biweekly rebalance. Novel curve segment — 10Y-2Y "
    "and 30Y-10Y already on leaderboard but mid-curve 5Y-3M belly untouched."
)

UNIVERSE = _universe

STRATEGY = BellyCurveFVX()
