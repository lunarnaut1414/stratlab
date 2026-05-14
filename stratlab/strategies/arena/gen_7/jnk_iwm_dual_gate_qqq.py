"""JNK Credit + IWM Small-Cap Dual Gate QQQ — gen_7 sonnet-7 (attempt 11)

Hypothesis: Hold QQQ when BOTH:
  1. JNK above its 50d SMA (credit conditions supportive)
  2. IWM 20d return > 0 (small caps leading = broad risk appetite)

When only JNK bullish (credit ok but breadth failing): SPY 97%
When neither: SHY 50% + TLT 47%

Rationale: Small-cap leadership (IWM outperforming) is a confirmed risk-on signal
that is distinct from the RSP/SPY breadth used above. When small caps are positive
on an absolute basis AND credit is supportive, the market's risk appetite is fully
engaged, making QQQ the best vehicle.

Key differences from existing strategies:
- gen6_jnk_vix_dual_gate_qqq: uses VIX level not IWM
- gen6_smallcap_leadership_rotation: uses IWM vs SPY relative (not absolute IWM)
  and doesn't combine with JNK MA gate in 3-tier fashion
- gen6_opus3_ensemble: complex ensemble, this is simpler direct signal

Weekly rebalance for >200 trades.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "SHY", "IWM", "JNK"]

REBALANCE_EVERY = 5        # weekly
JNK_MA = 50                # JNK 50d SMA
IWM_WINDOW = 20            # IWM 20d absolute return
TREND_WINDOW = 200
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_SHY = "SHY"
_IWM = "IWM"
_JNK = "JNK"


class JNKIWMDualGateQQQ(Strategy):
    """JNK 50d MA + IWM positive absolute return = QQQ; JNK only = SPY; else TLT+SHY."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        iwm_window: int = IWM_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            iwm_window=iwm_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.iwm_window = int(iwm_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.trend_window) + 10
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

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            if _SHY in live:
                target[_SHY] = 0.50 * self.exposure
            if _TLT in live:
                target[_TLT] = 0.47 * self.exposure
        else:
            need = max(self.jnk_ma, self.iwm_window) + 5
            prices = ctx.closes_window(need)

            # Signal 1: JNK above 50d SMA
            credit_bull = False
            if _JNK in prices.columns:
                jnk_col = prices[_JNK].dropna()
                if len(jnk_col) >= self.jnk_ma:
                    jnk_now = float(jnk_col.iloc[-1])
                    jnk_sma = float(jnk_col.iloc[-self.jnk_ma:].mean())
                    credit_bull = jnk_now > jnk_sma

            # Signal 2: IWM positive 20d absolute return
            smallcap_bull = False
            if _IWM in prices.columns:
                iwm_col = prices[_IWM].dropna()
                if len(iwm_col) >= self.iwm_window:
                    iwm_ret = float(iwm_col.iloc[-1] / iwm_col.iloc[-self.iwm_window] - 1.0)
                    smallcap_bull = np.isfinite(iwm_ret) and iwm_ret > 0

            if credit_bull and smallcap_bull:
                # Both bullish: QQQ
                if _QQQ in live:
                    target[_QQQ] = self.exposure
            elif credit_bull:
                # Credit only: SPY (stable but not growth-oriented)
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Credit weak: bonds
                if _SHY in live:
                    target[_SHY] = 0.50 * self.exposure
                if _TLT in live:
                    target[_TLT] = 0.47 * self.exposure

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


NAME = "jnk_iwm_dual_gate_qqq"
HYPOTHESIS = (
    "JNK 50d MA + IWM positive 20d return dual gate: both bullish -> QQQ 97%; "
    "JNK only -> SPY 97%; neither -> SHY+TLT; SPY 200d bear gate; weekly rebalance; "
    "distinct from JNK+VIX or RSP breadth gates by using absolute small-cap return "
    "as the risk appetite confirmation signal"
)

STRATEGY = JNKIWMDualGateQQQ()
