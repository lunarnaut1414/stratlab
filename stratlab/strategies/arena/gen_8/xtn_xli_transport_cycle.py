"""Industrials-vs-Staples economic cycle rotation (gen_8, sonnet-1).

Hypothesis:
    XLI (SPDR Industrials) vs XLP (SPDR Consumer Staples) 42-day return
    spread acts as an economic activity regime indicator. Cyclical
    industrials outperform when the economy is growing (demand for
    machinery, logistics, capital goods); defensive staples lead in
    contraction or uncertainty.

    * XLI 42d return > XLP 42d return (cyclicals lead) + SPY bull:
      -> hold QQQ 60% + IWM 37% (growth + small-cap benefit most)
    * XLP 42d return > XLI 42d return (defensives lead) + SPY bull:
      -> hold TLT 60% + IEF 37% (defensive rotation)
    * SPY below 200d SMA (bear):
      -> hold TLT 60% + IEF 37%

    Weekly rebalance. Industrials-vs-staples cycle regime with QQQ+IWM
    routing is not present in prior rounds — those used XLK/XLV, XLY/XLP,
    or VIX/credit signals.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
MOMENTUM_WINDOW = 42        # 42-day return for XLI/XLP comparison
TREND_WINDOW = 200          # SPY 200d SMA for bear gate
REBALANCE_EVERY = 5         # weekly rebalance
EXPOSURE = 0.97

NAME = "xtn_xli_transport_cycle"
HYPOTHESIS = (
    "XLI/XLP industrials-vs-staples 42d spread as economic-cycle signal: "
    "XLI outperforms -> hold QQQ 60%+IWM 37%; XLP outperforms or SPY bear "
    "-> TLT 60%+IEF 37%; weekly rebalance; industrials-vs-staples cycle "
    "routing to QQQ+IWM untouched on leaderboard"
)


class XLIXLPCycleRotation(Strategy):
    """Industrials-vs-staples 42d spread driving growth/defensive allocation."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.trend_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 1:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- XLI and XLP 42d return ---
        try:
            xli_hist = ctx.history("XLI")
            xlp_hist = ctx.history("XLP")
        except KeyError:
            return []

        need = self.momentum_window + 2
        xli_close = xli_hist["close"].dropna()
        xlp_close = xlp_hist["close"].dropna()
        if len(xli_close) < need or len(xlp_close) < need:
            return []

        xli_ret = float(xli_close.iloc[-1]) / float(xli_close.iloc[-(self.momentum_window + 1)]) - 1.0
        xlp_ret = float(xlp_close.iloc[-1]) / float(xlp_close.iloc[-(self.momentum_window + 1)]) - 1.0

        # --- Regime logic ---
        if not spy_bull or xlp_ret > xli_ret:
            # Defensive or bear: TLT 60% + IEF 37%
            target = {"TLT": 0.60, "IEF": 0.37}
        else:
            # Cyclicals leading: QQQ 60% + IWM 37%
            target = {"QQQ": 0.60, "IWM": 0.37}

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Buy/adjust target positions
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


STRATEGY = XLIXLPCycleRotation()
UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "IEF", "XLI", "XLP"]
