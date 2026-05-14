"""GLD/GDX relative strength regime momentum strategy.

Hypothesis: When gold miners (GDX) outperform the metal (GLD) on 20d return,
miners leading = risk-on commodity signal. In that regime, hold top-10 SP500
momentum stocks. When GLD leads (defensive gold = risk-off), hold GLD+TLT.
SPY 200d SMA gate provides secondary confirmation.

Rationale: Gold miners are leveraged plays on gold — they lead gold when
investors are risk-on (prefer operational leverage). When gold leads miners,
it signals fear-driven demand for the safety asset. This cross-asset signal
is orthogonal to VIX level, credit spreads, and yield curve signals already
on the leaderboard.

Distinction from existing strategies:
  - GDX/GLD relative strength as primary regime signal (not seen in any
    gen_5 or gen_6 strategy)
  - Routes to SP500 momentum in risk-on, GLD+TLT in risk-off
  - SPY 200d SMA as secondary gate
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 20    # 20d for GDX/GLD comparison
STOCK_MOM_WINDOW = 63   # 63d for stock momentum ranking
TOP_K = 10
TREND_WINDOW = 200
EXPOSURE = 0.97


class GldGdxRegimeMomentum(Strategy):
    """GLD vs GDX relative strength regime: risk-on -> SP500 momentum, risk-off -> GLD+TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stock_mom_window=stock_mom_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stock_mom_window = int(stock_mom_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_mom_window) + 10
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

        # Check SPY 200d SMA (secondary gate)
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

        # GDX vs GLD relative strength (primary regime signal)
        risk_on = True  # default to risk-on
        try:
            gld_hist = ctx.history("GLD")
            gdx_hist = ctx.history("GDX")
            if (gld_hist is not None and gdx_hist is not None
                    and len(gld_hist) >= self.momentum_window
                    and len(gdx_hist) >= self.momentum_window):
                gld_close = gld_hist["close"].dropna()
                gdx_close = gdx_hist["close"].dropna()
                if (len(gld_close) >= self.momentum_window
                        and len(gdx_close) >= self.momentum_window):
                    gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-self.momentum_window] - 1.0)
                    gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-self.momentum_window] - 1.0)
                    # miners lead = risk-on, gold leads = risk-off
                    risk_on = (gdx_ret > gld_ret)
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull_market or not risk_on:
            # Risk-off: GLD 60% + TLT 37%
            for sym, w in [("GLD", 0.60), ("TLT", 0.37)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: top-K SP500 stocks by 63d momentum
            need = self.stock_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.stock_mom_window:
                # fallback to SPY
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.stock_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.stock_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
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
    return sp500_tickers() + ["GLD", "GDX", "TLT", "SPY"]


NAME = "gld_gdx_regime_momentum"
HYPOTHESIS = (
    "GLD/GDX relative strength regime: when GDX outperforms GLD on 20d return "
    "(miners leading metal = risk-on commodity signal) hold top-10 SP500 momentum stocks; "
    "when GLD leads hold GLD 60%+TLT 37%; biweekly rebalance with SPY 200d SMA gate"
)

UNIVERSE = _universe

STRATEGY = GldGdxRegimeMomentum()
