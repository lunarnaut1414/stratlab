"""VVIX-regime QQQ/SPY/TLT allocator.

Hypothesis: Use ^VVIX (VIX-of-VIX, vol-of-vol) relative to its 90-day MA
as a regime signal. When vol-of-vol is calm (VVIX < 90d MA) AND SPY is in
an uptrend (above 200d SMA), the market is in a stable expansion phase:
hold QQQ (higher beta/growth). When VVIX is elevated but SPY is still
bullish, shift to SPY (lower beta). When SPY is bearish, hold TLT.

Rationale: VVIX measures realized vol of the VIX itself. When VVIX is
below its moving average, the volatility regime is stable — even if VIX
is elevated, the uncertainty about uncertainty is low, suggesting the risk
environment is predictable. This is different from VIX level (which
measures expected vol) and credit spread (which measures default risk).

Key structural differences from existing strategies:
- Uses VVIX/^VVIX (vol-of-vol) not VIX level or credit signal
- VVIX vs its own 90d MA (relative, not absolute threshold)
- QQQ vs SPY tilt based on VVIX regime (not credit or breadth)
- Different from qqq_bollinger_vvix_dipbuy (which uses VVIX for dip-buy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "TLT", "^VVIX"]

VVIX_MA_PERIOD = 90   # 90d MA for VVIX regime
TREND_WINDOW = 200    # SPY 200d SMA
REBALANCE_EVERY = 5   # weekly
EXPOSURE = 0.97


class VvixRegimeQqqAllocator(Strategy):
    def __init__(
        self,
        vvix_ma_period: int = VVIX_MA_PERIOD,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            vvix_ma_period=vvix_ma_period,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.vvix_ma_period = int(vvix_ma_period)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vvix_ma_period, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY trend filter
        spy_bullish = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_now = float(spy_close.iloc[-1])
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bullish = spy_now > spy_sma
        except KeyError:
            pass

        # VVIX regime signal
        vvix_calm = False
        try:
            vvix_hist = ctx.history("^VVIX")
            if vvix_hist is not None and len(vvix_hist) >= self.vvix_ma_period + 2:
                vvix_close = vvix_hist["close"].dropna()
                if len(vvix_close) >= self.vvix_ma_period + 1:
                    vvix_now = float(vvix_close.iloc[-1])
                    vvix_ma = float(vvix_close.iloc[-self.vvix_ma_period:].mean())
                    vvix_calm = (vvix_now < vvix_ma) and np.isfinite(vvix_now) and np.isfinite(vvix_ma)
        except KeyError:
            pass

        # Determine target allocation
        target: dict[str, float] = {}

        if not spy_bullish:
            # Bear market: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = 0.50 * self.exposure
        elif vvix_calm:
            # Bull + stable vol-of-vol: aggressive QQQ
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Bull + elevated vol-of-vol: moderate SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure

        if not target and "SPY" in live:
            target["SPY"] = self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "vvix_regime_qqq_allocator"
HYPOTHESIS = (
    "VVIX regime equity allocator: rank exposure aggressively when vol-of-vol (VVIX) "
    "is below 90d MA (market calm and stable); hold QQQ 97% when VVIX<90d MA AND "
    "SPY>200d SMA; SPY 97% when VVIX>90d MA but SPY bullish; TLT 97% when "
    "SPY<200d SMA; weekly rebalance"
)

STRATEGY = VvixRegimeQqqAllocator()
