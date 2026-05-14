"""SMH vs IEF growth-vs-defensives gate for QQQ allocation (gen_8, sonnet-1).

Hypothesis:
    When semiconductors (SMH) are outperforming intermediate bonds (IEF) on
    a 63-day basis, the growth/risk-on environment is intact and QQQ is
    optimal. When bonds outperform semis, growth is decelerating or risk is
    elevated — hold SPY (reduced beta) or TLT (bear market).

    This is a fundamentally different signal from JNK/LQD credit spreads or
    VIX level — it uses actual sector return competition between the highest-
    growth sector (semis) and the safest-haven sector (bonds) as a regime
    discriminator.

    * SMH 63d return > IEF 63d return + SPY > 200d SMA:
      -> semiconductor/growth cycle intact -> QQQ 97%
    * IEF 63d return > SMH 63d return + SPY > 200d SMA:
      -> bonds outperforming semis in bull market (growth cooling) -> SPY 97%
    * SPY < 200d SMA (bear):
      -> TLT 60% + IEF 37%

    Biweekly rebalance. SMH vs IEF cross-asset competition as QQQ/SPY
    timing mechanism is absent from all prior rounds.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
MOMENTUM_WINDOW = 63        # 63-day (3-month) return comparison
SPY_TREND = 200             # SPY 200d SMA bear gate
REBALANCE_EVERY = 10        # biweekly
EXPOSURE = 0.97

NAME = "smh_ief_growth_gate"
HYPOTHESIS = (
    "SMH vs IEF 63d return comparison as growth regime gate: SMH outperforms IEF "
    "+ SPY bull -> QQQ 97%; IEF outperforms SMH + SPY bull -> SPY 97%; "
    "SPY bear -> TLT 60%+IEF 37%; biweekly rebalance; semis-vs-bonds competition "
    "as novel growth-gate for QQQ/SPY routing absent from all prior rounds"
)


class SMHIEFGrowthGate(Strategy):
    """SMH vs IEF 63d return drives QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend: int = SPY_TREND,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            spy_trend=spy_trend,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.spy_trend = int(spy_trend)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend) + 5
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

        # --- SMH and IEF 63d return ---
        need = self.momentum_window + 2
        try:
            smh_hist = ctx.history("SMH")
            ief_hist = ctx.history("IEF")
        except KeyError:
            return []

        smh_close = smh_hist["close"].dropna()
        ief_close = ief_hist["close"].dropna()
        if len(smh_close) < need or len(ief_close) < need:
            return []

        smh_ret = float(smh_close.iloc[-1]) / float(smh_close.iloc[-(self.momentum_window + 1)]) - 1.0
        ief_ret = float(ief_close.iloc[-1]) / float(ief_close.iloc[-(self.momentum_window + 1)]) - 1.0

        # --- Regime routing ---
        if not spy_bull:
            # Bear market -> defensive bonds
            target = {"TLT": 0.60, "IEF": 0.37}
        elif smh_ret > ief_ret:
            # Semis outperforming bonds -> growth regime -> QQQ
            target = {"QQQ": self.exposure}
        else:
            # Bonds outperforming semis in bull -> caution, hold SPY
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


STRATEGY = SMHIEFGrowthGate()
UNIVERSE = ["SPY", "QQQ", "SMH", "IEF", "TLT"]
