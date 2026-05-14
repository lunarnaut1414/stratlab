"""Dual-trend bond+equity simultaneous confirmation regime (gen_8, sonnet-1).

Hypothesis:
    Use BOTH SPY and TLT trend states simultaneously as a 2-dimensional
    regime classifier. Specifically:
    - When TLT is trending UP (bonds rallying, rates falling) AND SPY is bullish:
      growth regime with rate tailwind -> QQQ 97% (tech/growth benefits from falling rates)
    - When TLT is trending DOWN (bonds selling, rates rising) AND SPY is bullish:
      reflation / late-cycle regime -> SPY 60% + XLF 37% (financials benefit from rising rates)
    - When SPY is bearish AND TLT bull (flight-to-safety):
      -> TLT 97%
    - When SPY is bearish AND TLT bear (stagflation):
      -> IEF 60% + SHY 37%

    Weekly rebalance. Uses TLT trend direction to select WHICH equity vehicle
    to hold (not whether to hold equity), producing distinct return path from
    all prior strategies. XLF tilt in reflation regime is not on leaderboard.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
SPY_TREND = 200     # SPY trend window (200d SMA)
TLT_TREND = 63      # TLT trend window (63d SMA)
REBALANCE_EVERY = 5 # weekly (5 bars)
EXPOSURE = 0.97

NAME = "dual_trend_bond_equity"
HYPOTHESIS = (
    "Dual-trend bond+equity regime v2: SPY bull+TLT bull (falling rates) -> QQQ 97%; "
    "SPY bull+TLT bear (rising rates/reflation) -> SPY 60%+XLF 37%; "
    "SPY bear+TLT bull -> TLT 97%; SPY bear+TLT bear -> IEF 60%+SHY 37%; "
    "weekly rebalance; TLT direction selects equity vehicle not equity/bond split"
)


class DualTrendBondEquity(Strategy):
    """TLT direction selects QQQ vs SPY+XLF in equity regime; TLT/IEF in bear."""

    def __init__(
        self,
        spy_trend: int = SPY_TREND,
        tlt_trend: int = TLT_TREND,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            spy_trend=spy_trend,
            tlt_trend=tlt_trend,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.spy_trend = int(spy_trend)
        self.tlt_trend = int(tlt_trend)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend, self.tlt_trend) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- TLT trend ---
        try:
            tlt_hist = ctx.history("TLT")
        except KeyError:
            return []
        tlt_close = tlt_hist["close"].dropna()
        if len(tlt_close) < self.tlt_trend:
            return []
        tlt_sma = float(tlt_close.iloc[-self.tlt_trend:].mean())
        tlt_bull = float(tlt_close.iloc[-1]) > tlt_sma

        # --- 2x2 regime routing ---
        if spy_bull and tlt_bull:
            # Equity bull + falling rates -> QQQ (growth / duration benefit)
            target = {"QQQ": self.exposure}
        elif spy_bull and not tlt_bull:
            # Reflation / rising rates + equity bull -> SPY + financials tilt
            target = {"SPY": 0.60, "XLF": 0.37}
        elif not spy_bull and tlt_bull:
            # Classic flight-to-safety -> max duration TLT
            target = {"TLT": self.exposure}
        else:
            # Both falling -> mid-duration + cash
            target = {"IEF": 0.60, "SHY": 0.37}

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity_val = ctx.portfolio_value(live)
        if equity_val <= 0:
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
            tgt_shares = int(equity_val * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


STRATEGY = DualTrendBondEquity()
UNIVERSE = ["SPY", "QQQ", "TLT", "IEF", "SHY", "XLF"]
