"""SOXX vs IWM quality-vs-speculative regime (gen_8, sonnet-1).

Hypothesis:
    SOXX (iShares Semiconductor ETF) vs IWM (Russell 2000 small-caps) 42-day
    relative strength as a quality/risk-appetite indicator.

    Semiconductors are large-cap, high-margin, global-cycle businesses.
    Small-caps (IWM) represent broader domestic risk appetite. When SOXX
    outperforms IWM, the market is in a quality-growth regime favorable to QQQ.
    When IWM outperforms SOXX, broader risk appetite is elevated and small-caps
    are in favor.

    * SOXX 42d return > IWM 42d return + SPY bull + both positive:
      -> quality semiconductor cycle -> QQQ 97%
    * IWM 42d return > SOXX 42d return + SPY bull + both positive:
      -> speculative small-cap regime -> SPY 60% + IWM 37%
    * Either negative or SPY bear:
      -> defensive -> TLT 60% + IEF 37%

    Weekly rebalance. SOXX vs IWM as quality/risk-appetite proxy driving
    QQQ vs SPY+IWM is absent from all prior rounds. gen7_opus2_smh_kbe
    uses SMH vs KBE (banks) differently.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
MOMENTUM_WINDOW = 42        # 42-day return
SPY_TREND = 200             # SPY 200d SMA bear gate
REBALANCE_EVERY = 5         # weekly
EXPOSURE = 0.97

NAME = "soxx_iwm_quality_signal"
HYPOTHESIS = (
    "SOXX/IWM semiconductor-vs-small-cap 42d spread as quality-regime signal: "
    "SOXX leads + both positive + SPY bull -> QQQ 97%; IWM leads + both positive "
    "+ SPY bull -> SPY 60%+IWM 37%; either negative or SPY bear -> TLT 60%+IEF 37%; "
    "weekly rebalance; SOXX vs IWM quality proxy distinct from SMH/KBE prior strategies"
)


class SOXXIWMQualitySignal(Strategy):
    """SOXX vs IWM 42d spread drives QQQ/SPY+IWM/TLT allocation."""

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

        # --- SOXX and IWM 42d return ---
        need = self.momentum_window + 2
        try:
            soxx_hist = ctx.history("SOXX")
            iwm_hist = ctx.history("IWM")
        except KeyError:
            return []

        soxx_close = soxx_hist["close"].dropna()
        iwm_close = iwm_hist["close"].dropna()
        if len(soxx_close) < need or len(iwm_close) < need:
            return []

        soxx_ret = float(soxx_close.iloc[-1]) / float(soxx_close.iloc[-(self.momentum_window + 1)]) - 1.0
        iwm_ret = float(iwm_close.iloc[-1]) / float(iwm_close.iloc[-(self.momentum_window + 1)]) - 1.0

        # --- Regime routing ---
        if not spy_bull or (soxx_ret < 0 and iwm_ret < 0):
            # Bear or both negative -> defensive bonds
            target = {"TLT": 0.60, "IEF": 0.37}
        elif soxx_ret < 0 or iwm_ret < 0:
            # One negative -> partial defensive
            target = {"TLT": 0.60, "IEF": 0.37}
        elif soxx_ret > iwm_ret:
            # Semis outperform (quality cycle) -> growth QQQ
            target = {"QQQ": self.exposure}
        else:
            # Small-caps outperform -> broader risk-on -> SPY + IWM
            target = {"SPY": 0.60, "IWM": 0.37}

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


STRATEGY = SOXXIWMQualitySignal()
UNIVERSE = ["SPY", "QQQ", "SOXX", "IWM", "TLT", "IEF"]
