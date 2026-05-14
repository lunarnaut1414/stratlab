"""Software-sector relative strength as QQQ timing signal (gen_8, sonnet-1).

Hypothesis:
    IGV (iShares Expanded Tech-Software ETF) vs SPY 21-day relative return
    as a QQQ rotation signal. Software leads broad-market tops and bottoms —
    when software is outperforming SPY, tech/growth is in favor and QQQ
    benefits. When software underperforms, rotate to SPY or defensives.

    * IGV 21d return > SPY 21d return + SPY bull (200d SMA) + IGV 21d > 0:
      -> software leadership -> QQQ 97%
    * SPY bull but IGV underperforms or IGV negative:
      -> broad equity without growth premium -> SPY 97%
    * SPY bear (SPY < 200d SMA):
      -> defensive -> TLT 60% + IEF 37%

    Biweekly rebalance (every 10 bars). IGV vs SPY relative strength as
    QQQ timing mechanism is absent from all prior rounds.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
REL_WINDOW = 21             # 21-day return for IGV vs SPY relative strength
SPY_TREND = 200             # SPY 200d SMA bear gate
REBALANCE_EVERY = 10        # biweekly
EXPOSURE = 0.97

NAME = "igv_software_qqq_timing"
HYPOTHESIS = (
    "IGV software ETF vs SPY 21d relative strength as QQQ timing signal: "
    "IGV > SPY + SPY bull -> QQQ 97%; SPY bull but IGV trails -> SPY 97%; "
    "SPY bear -> TLT 60%+IEF 37%; biweekly rebalance; software-relative-strength "
    "QQQ timing absent from all prior rounds"
)


class IGVSoftwareQQQTiming(Strategy):
    """IGV vs SPY 21d relative strength drives QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        rel_window: int = REL_WINDOW,
        spy_trend: int = SPY_TREND,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rel_window=rel_window,
            spy_trend=spy_trend,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.rel_window = int(rel_window)
        self.spy_trend = int(spy_trend)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.rel_window, self.spy_trend) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- IGV vs SPY 21d return ---
        need = self.rel_window + 2
        try:
            igv_hist = ctx.history("IGV")
        except KeyError:
            return []

        igv_close = igv_hist["close"].dropna()
        if len(igv_close) < need or len(spy_close) < need:
            return []

        igv_ret = float(igv_close.iloc[-1]) / float(igv_close.iloc[-(self.rel_window + 1)]) - 1.0
        spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-(self.rel_window + 1)]) - 1.0

        # --- Regime routing ---
        if not spy_bull:
            # Bear market -> defensive
            target = {"TLT": 0.60, "IEF": 0.37}
        elif igv_ret > spy_ret and igv_ret > 0:
            # Software outperforming with positive return -> growth leader -> QQQ
            target = {"QQQ": self.exposure}
        else:
            # Software trailing or negative -> broad equity without growth premium
            target = {"SPY": self.exposure}

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


STRATEGY = IGVSoftwareQQQTiming()
UNIVERSE = ["SPY", "QQQ", "IGV", "TLT", "IEF"]
