"""SMH semiconductors vs XLU utilities cycle regime (gen_8, sonnet-1).

Hypothesis:
    SMH (VanEck Semiconductor ETF) vs XLU (SPDR Utilities) 42-day relative
    strength as a macro risk-appetite and economic-cycle indicator.
    Semiconductors are the most cyclical/growth-dependent sector;
    utilities are the most defensive. Their relative momentum captures
    the risk-on/risk-off spectrum with high sensitivity.

    * SMH 42d return > XLU 42d return + SPY bull (semiconductor leadership):
      -> aggressive growth regime -> QQQ 97%
    * XLU 42d return > SMH 42d return + SPY bull (defensive leadership):
      -> defensive within equity -> SPY 60% + XLU 37%
    * Both negative OR SPY bear:
      -> defensive -> TLT 60% + IEF 37%

    Weekly rebalance. SMH/XLU relative strength as cross-sector cycle
    proxy for QQQ vs SPY+XLU allocation is not present in any prior round.
    The existing gen7_opus2_smh_kbe_industry_leadership uses SMH vs KBE
    (banks) differently and without utilities angle.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
MOMENTUM_WINDOW = 42        # 42-day return for SMH/XLU comparison
SPY_TREND = 200             # SPY 200d SMA bear gate
REBALANCE_EVERY = 5         # weekly rebalance
EXPOSURE = 0.97

NAME = "biotech_healthcare_cycle"
HYPOTHESIS = (
    "SMH/XLU semiconductor-vs-utilities 42d spread as growth/defensive cycle signal: "
    "SMH leads -> QQQ 97%; XLU leads + both positive -> SPY 60%+XLU 37%; "
    "both negative or SPY bear -> TLT 60%+IEF 37%; weekly rebalance; "
    "SMH vs utilities cycle proxy distinct from prior SMH/KBE or VIX signals"
)


class SMHXLUCycleRotation(Strategy):
    """SMH vs XLU 42d spread drives QQQ/SPY+XLU/TLT allocation."""

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

        # --- SMH and XLU 42d return ---
        need = self.momentum_window + 2
        try:
            smh_hist = ctx.history("SMH")
            xlu_hist = ctx.history("XLU")
        except KeyError:
            return []

        smh_close = smh_hist["close"].dropna()
        xlu_close = xlu_hist["close"].dropna()
        if len(smh_close) < need or len(xlu_close) < need:
            return []

        smh_ret = float(smh_close.iloc[-1]) / float(smh_close.iloc[-(self.momentum_window + 1)]) - 1.0
        xlu_ret = float(xlu_close.iloc[-1]) / float(xlu_close.iloc[-(self.momentum_window + 1)]) - 1.0

        # --- Regime routing ---
        if not spy_bull or (smh_ret < 0 and xlu_ret < 0):
            # Bear or both negative -> defensive bonds
            target = {"TLT": 0.60, "IEF": 0.37}
        elif smh_ret > xlu_ret:
            # Semis lead (risk-on cycle signal) -> growth tilt QQQ
            target = {"QQQ": self.exposure}
        else:
            # Utilities lead (defensive within equity) -> SPY + XLU
            target = {"SPY": 0.60, "XLU": 0.37}

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


STRATEGY = SMHXLUCycleRotation()
UNIVERSE = ["SPY", "QQQ", "SMH", "XLU", "TLT", "IEF"]
