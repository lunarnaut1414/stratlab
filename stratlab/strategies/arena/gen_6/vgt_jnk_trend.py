"""VGT tech ETF with JNK credit gate — gen_6 sonnet-7

Hypothesis: Hold VGT (Vanguard Information Technology ETF) when JNK is
above its 45d SMA (credit healthy) AND SPY is above its 150d SMA (equity
bull). Hold QQQ when SPY is bullish but credit is weak (tech overweight
preserved but with broader index safety). Hold TLT when SPY below 150d SMA.
Rebalance every 5 bars.

Rationale:
  VGT (pure tech sector) outperformed QQQ in 2010-2018 because it has
  concentrated technology exposure without QQQ's healthcare/consumer names.
  Using JNK 45d SMA (different MA length) as credit gate and SPY 150d SMA
  (medium-term trend) creates a different timing pattern from existing
  strategies that use JNK 20d, 30d, or 60d MA.

  Distinct from existing leaderboard:
  - VGT as primary equity (not QQQ or SPY) — concentrated tech
  - JNK 45d SMA (different from 20d/30d/60d used elsewhere)
  - QQQ as intermediate state (not SPY) when credit weakens
  - SPY 150d SMA (between 100d and 200d used elsewhere)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5    # weekly
JNK_MA = 45            # JNK 45d SMA (unique length)
TREND_WINDOW = 150     # SPY 150d SMA
EXPOSURE = 0.97


class VGTJNKTrend(Strategy):
    """VGT/QQQ/TLT rotation with JNK 45d SMA + SPY 150d SMA."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        jnk_ma: int = JNK_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            jnk_ma=jnk_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.jnk_ma = int(jnk_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.jnk_ma, self.trend_window) + 5
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

        # SPY trend gate (150d SMA)
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # JNK credit health (45d SMA)
        jnk_healthy = True
        try:
            jnk_hist = ctx.history("JNK")
            if len(jnk_hist) >= self.jnk_ma + 2:
                jnk_close = jnk_hist["close"].dropna().values
                jnk_sma = float(np.mean(jnk_close[-self.jnk_ma:]))
                jnk_healthy = float(jnk_close[-1]) > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif jnk_healthy:
            # Credit healthy + bull: VGT (tech sector)
            if "VGT" in closes_now.index:
                target["VGT"] = self.exposure
        else:
            # Credit weak but SPY in trend: QQQ (broader tech)
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


NAME = "vgt_jnk_trend"
HYPOTHESIS = (
    "VGT tech ETF with JNK 45d SMA credit gate: JNK>45d SMA AND SPY>150d SMA → VGT 97%; "
    "JNK weak but SPY bull → QQQ 97%; SPY bear → TLT 97%; weekly rebalance; "
    "concentrated tech (VGT) with unique 45d JNK and 150d SPY timing"
)
UNIVERSE = ["VGT", "QQQ", "TLT", "JNK", "SPY"]
STRATEGY = VGTJNKTrend()
