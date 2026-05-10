"""HYG/IEF credit-spread ratio momentum with SPY trend strategy.

Hypothesis: Use the HYG/IEF price ratio as a high-yield credit spread proxy.
When the ratio's 20d MA > 60d MA (credit spreads tightening = risk-on) AND
SPY is above its 200d SMA, hold QQQ 97%.
When the ratio is trending flat/neutral but SPY still bullish, hold SPY 97%.
When the ratio's 20d MA < 60d MA (spreads widening = risk-off), hold TLT 97%.

Rationale: HYG/IEF ratio (high-yield / investment-grade treasury) is a
cleaner credit risk signal than raw JNK price because it isolates the credit
spread component from duration risk. Both HYG and IEF move together when
rates change, but their ratio captures pure credit risk appetite.

Structural differences from existing credit strategies:
- Uses ratio HYG/IEF (not raw JNK price or JNK/LQD ratio)
- 3-tier: QQQ (risk-on) / SPY (neutral) / TLT (risk-off)
- 20d vs 60d MA on the ratio - medium-term credit trend
- Combines credit ratio with SPY 200d trend (equity confirmation)
- Different from: hy_credit_qqq_rotation (JNK level), jnk_lqd_spy_regime (JNK/LQD ratio)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["HYG", "IEF", "QQQ", "SPY", "TLT"]

FAST_MA = 20
SLOW_MA = 60
TREND_WINDOW = 200    # SPY 200d SMA
REBALANCE_EVERY = 5   # weekly
EXPOSURE = 0.97
# Threshold for "significant" ratio divergence
THRESHOLD = 0.001     # 0.1% relative spread between fast/slow MA


class HygIefRatioRotation(Strategy):
    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
        threshold: float = THRESHOLD,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
            threshold=threshold,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self.threshold = float(threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.trend_window) + 5
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

        # HYG/IEF ratio signal
        credit_strong = False
        credit_weak = False
        try:
            hyg_hist = ctx.history("HYG")
            ief_hist = ctx.history("IEF")
            if (hyg_hist is not None and len(hyg_hist) >= self.slow_ma + 2 and
                    ief_hist is not None and len(ief_hist) >= self.slow_ma + 2):
                hyg_close = hyg_hist["close"].dropna()
                ief_close = ief_hist["close"].dropna()

                # Align lengths
                min_len = min(len(hyg_close), len(ief_close))
                hyg_c = hyg_close.iloc[-min_len:]
                ief_c = ief_close.iloc[-min_len:]

                if min_len >= self.slow_ma and (ief_c > 0).all():
                    ratio = hyg_c.values / ief_c.values
                    fast_val = float(np.mean(ratio[-self.fast_ma:]))
                    slow_val = float(np.mean(ratio[-self.slow_ma:]))
                    if slow_val > 0:
                        rel_diff = (fast_val - slow_val) / slow_val
                        if rel_diff > self.threshold:
                            credit_strong = True
                        elif rel_diff < -self.threshold:
                            credit_weak = True
        except KeyError:
            pass

        # Determine target allocation
        target: dict[str, float] = {}

        if credit_weak:
            # Spreads widening: risk-off → TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
            elif "IEF" in live:
                target["IEF"] = self.exposure
        elif credit_strong and spy_bullish:
            # Spreads tightening + bull market: QQQ aggressive
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        elif spy_bullish:
            # Neutral credit + bull market: SPY
            if "SPY" in live:
                target["SPY"] = self.exposure
        else:
            # Credit neutral + bear market: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = 0.60 * self.exposure

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


NAME = "hyg_ief_ratio_rotation"
HYPOTHESIS = (
    "HYG/IEF spread ratio momentum with SPY trend: use 20d MA vs 60d MA of HYG/IEF "
    "price ratio as credit-spread proxy; when ratio trending up AND SPY>200d SMA hold "
    "QQQ 97%; when ratio flat AND SPY trending hold SPY 97%; when ratio falling hold "
    "TLT 97%; weekly rebalance; ratio signal differs from raw JNK level used by "
    "existing strategies"
)

STRATEGY = HygIefRatioRotation()
