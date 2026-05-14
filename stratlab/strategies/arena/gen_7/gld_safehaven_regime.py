"""GLD Trend as Macro Safe-Haven Signal — gen_7 sonnet-3

Hypothesis: Gold is a safe-haven asset. When GLD 20d return > 2% (gold
surging = risk-off flight to safety), rotate to TLT 60%+GLD 37%.
When GLD 20d return < -1% (gold falling = risk-on), hold QQQ 97%.
Else hold SPY 97%. SPY 200d SMA bear override to TLT. Weekly rebalance.

Rationale: Gold surges during flight-to-safety periods (credit stress,
geopolitical risk, stagflation). A sharp GLD rally signals macro risk-off
before other indicators react. Conversely, falling GLD suggests comfortable
risk appetite, favoring growth equities (QQQ). This is orthogonal to VIX-level,
credit-spread, and breadth signals, which are the dominant leaderboard signals.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
GLD_WINDOW = 20            # 20d return for gold signal
RISK_OFF_THRESHOLD = 0.02  # GLD +2% over 20d -> risk-off
RISK_ON_THRESHOLD = -0.01  # GLD -1% over 20d -> risk-on
TREND_WINDOW = 200         # SPY 200d SMA
EXPOSURE = 0.97


class GldSafehavenRegime(Strategy):
    """GLD 20d return as macro safe-haven regime signal."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        gld_window: int = GLD_WINDOW,
        risk_off_threshold: float = RISK_OFF_THRESHOLD,
        risk_on_threshold: float = RISK_ON_THRESHOLD,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            gld_window=gld_window,
            risk_off_threshold=risk_off_threshold,
            risk_on_threshold=risk_on_threshold,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.gld_window = int(gld_window)
        self.risk_off_threshold = float(risk_off_threshold)
        self.risk_on_threshold = float(risk_on_threshold)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.gld_window) + 5
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
        spy_sma200 = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        # GLD 20d return signal
        try:
            gld_hist = ctx.history("GLD")
        except KeyError:
            return []
        if len(gld_hist) < self.gld_window + 5:
            return []
        gld_close = gld_hist["close"].dropna()
        if len(gld_close) < self.gld_window + 1:
            return []
        gld_last = float(gld_close.iloc[-1])
        gld_start = float(gld_close.iloc[-self.gld_window])
        if gld_start <= 0 or not np.isfinite(gld_start) or not np.isfinite(gld_last):
            return []
        gld_ret_20d = gld_last / gld_start - 1.0

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif gld_ret_20d > self.risk_off_threshold:
            # Gold surging: risk-off, rotate to TLT+GLD
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure * 0.60
            if "GLD" in closes_now.index:
                target["GLD"] = self.exposure * 0.37
        elif gld_ret_20d < self.risk_on_threshold:
            # Gold falling: risk-on, hold QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
            elif "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            # Neutral: hold SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

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


UNIVERSE = ["SPY", "QQQ", "TLT", "GLD"]

NAME = "gld_safehaven_regime"
HYPOTHESIS = (
    "GLD trend as macro safe-haven signal: when GLD 20d return > 2% (gold surging, risk-off), "
    "rotate to TLT 60%+GLD 37%; when GLD 20d return < -1% (gold falling, risk-on) hold QQQ 97%; "
    "else hold SPY 97%; SPY 200d SMA bear override to TLT; weekly rebalance"
)

STRATEGY = GldSafehavenRegime()
