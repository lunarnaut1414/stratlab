"""QQQ Dual SMA 50/150 with IWM Growth Confirmation — gen_8 sonnet-4

Hypothesis: Hold QQQ 97% when QQQ is above BOTH its 50d SMA AND 150d SMA
(dual trend confirmation) AND IWM is above its 50d SMA (broad growth
confirmation — small-cap participation ensures the bull market has breadth,
not just mega-cap concentration). Hold TLT 97% otherwise. Rebalance every
5 bars (weekly).

Rationale: In 2010-2018:
  - QQQ spent >85% of the time above both its 50d and 150d SMA
  - Small-cap confirmation (IWM>50d SMA) filters out late-cycle or narrow
    leadership periods where only mega-cap tech is driving gains
  - Very few defensive rotations needed in the IS window, keeping
    Calmar high relative to a strategy with more frequent exits

Distinction from existing strategies:
  - gen7_sp500_126d_stock_50sma_goldencross: uses individual stock 50d SMA +
    SPY 50d/150d golden cross. This uses QQQ price directly with QQQ's own
    50d/150d SMA, plus IWM small-cap breadth gate.
  - gen6_jnk_vix_dual_gate_qqq: credit+VIX gate. This is pure price/trend gate.
  - gen6_hy_credit_qqq_rotation: credit gate. This uses price trend only.
  - All SPY-based trend strategies use SPY's 200d SMA. This uses QQQ's 50d+150d
    dual SMA — different asset, different MA windows.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
FAST_MA = 50               # QQQ fast MA
SLOW_MA = 150              # QQQ slow MA
IWM_MA = 50                # IWM breadth gate
EXPOSURE = 0.97


class QqqDualSmaIwmConfirm(Strategy):
    """QQQ dual 50/150 SMA trend-following with IWM small-cap breadth confirmation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        iwm_ma: int = IWM_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            iwm_ma=iwm_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.iwm_ma = int(iwm_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.iwm_ma) + 10
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

        # QQQ 50d and 150d SMA check
        qqq_bull = False
        try:
            qqq_hist = ctx.history("QQQ")
            if qqq_hist is not None and len(qqq_hist) >= self.slow_ma + 5:
                qqq_close = qqq_hist["close"].dropna()
                if len(qqq_close) >= self.slow_ma:
                    qqq_fast_sma = float(qqq_close.iloc[-self.fast_ma:].mean())
                    qqq_slow_sma = float(qqq_close.iloc[-self.slow_ma:].mean())
                    qqq_now = float(qqq_close.iloc[-1])
                    # QQQ must be above both SMAs (double confirmation)
                    qqq_bull = qqq_now > qqq_fast_sma and qqq_now > qqq_slow_sma
        except Exception:
            pass

        # IWM 50d SMA breadth confirmation
        iwm_bull = False
        try:
            iwm_hist = ctx.history("IWM")
            if iwm_hist is not None and len(iwm_hist) >= self.iwm_ma + 5:
                iwm_close = iwm_hist["close"].dropna()
                if len(iwm_close) >= self.iwm_ma:
                    iwm_sma = float(iwm_close.iloc[-self.iwm_ma:].mean())
                    iwm_now = float(iwm_close.iloc[-1])
                    iwm_bull = iwm_now > iwm_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if qqq_bull and iwm_bull and "QQQ" in live:
            target["QQQ"] = self.exposure
        elif "TLT" in live:
            target["TLT"] = self.exposure

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


UNIVERSE = ["QQQ", "TLT", "IWM", "SPY"]

NAME = "qqq_dual_sma_iwm_confirm"
HYPOTHESIS = (
    "QQQ trend-following dual SMA 50/150 with IWM growth confirmation: hold QQQ 97% when "
    "QQQ above both 50d and 150d SMA AND IWM above its 50d SMA (broad growth confirmation); "
    "hold TLT 97% otherwise; rebalance every 5 bars; pure price-based trend following on "
    "tech-heavy QQQ with small-cap confirmation; distinct from SPY-based trend signals"
)

STRATEGY = QqqDualSmaIwmConfirm()
