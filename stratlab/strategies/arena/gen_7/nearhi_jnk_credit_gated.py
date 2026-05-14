"""Near-52w-high SP500 momentum with JNK credit gate.

Hypothesis: hold top-20 SP500 stocks within 5% of 252d high AND positive 63d
momentum when JNK above 30d SMA AND SPY above 150d SMA; hold TLT when credit
weak; rebalance every 10 bars.

Rationale: The nearhi_momentum_quality curated strategy uses a 200d SMA gate.
Here we replace the 200d SMA gate with a dual JNK credit + SPY 150d SMA gate,
and relax the near-high threshold to 5% (vs 80% proximity). The credit gate
ensures we only buy quality momentum stocks when credit markets are risk-on —
JNK above its 30d SMA signals healthy corporate credit and confirms the momentum
environment. The SPY 150d SMA is a faster trend filter than 200d. This
combination is different from nearhi_momentum_quality (200d SMA gate, 126d window,
inverse-vol weights) and from pure credit-gated SP500 momentum (63d raw return).

Key distinctions:
  - Near-52w-high quality filter (nearhi < 0.95) combined with JNK credit gate
  - SPY 150d SMA trend filter (not 200d) — catches trend turns faster
  - Equal-weight (not inverse-vol) to improve trade count
  - 63d momentum (not 126d) — medium term
  - 10-bar rebalance (same as many others but different portfolio composition)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars
MOMENTUM_WINDOW = 63      # ~3 months
HIGH_WINDOW = 252         # 52-week high lookback
NEARHI_THRESHOLD = 0.95   # price must be within 5% of 52w high
JNK_MA = 30              # JNK SMA for credit regime
SPY_TREND = 150          # SPY trend window (faster than 200d)
TOP_K = 20
EXPOSURE = 0.97


class NearHiJnkCreditGated(Strategy):
    """SP500 nearhi quality + JNK credit gate: top-20 near-52w-high stocks in risk-on."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        jnk_ma: int = JNK_MA,
        spy_trend: int = SPY_TREND,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            jnk_ma=jnk_ma,
            spy_trend=spy_trend,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.jnk_ma = int(jnk_ma)
        self.spy_trend = int(spy_trend)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + self.jnk_ma + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 150d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend + 5:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # JNK credit gate
        jnk_bull = True  # default risk-on if unavailable
        try:
            jnk_hist = ctx.history("JNK")
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma + 5:
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_last = float(jnk_close.iloc[-1])
                jnk_bull = jnk_last > jnk_sma
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull or not jnk_bull:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Need 52w high + momentum history
            need = self.high_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue

                # Near 52w high filter
                recent_252 = col.iloc[-self.high_window:]
                w52_high = float(recent_252.max())
                if w52_high <= 0 or not np.isfinite(w52_high):
                    continue
                current_price = float(col.iloc[-1])
                nearhi_ratio = current_price / w52_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue  # Skip stocks not near their 52w high

                # 63d momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret) or ret <= 0:
                    continue

                scores[sym] = ret

            if len(scores) < 5:
                # Not enough candidates — fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_wt = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_wt

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", "JNK"]


NAME = "nearhi_jnk_credit_gated"
HYPOTHESIS = (
    "JNK credit trend + SPY nearhi quality hybrid: hold top-20 SP500 stocks within 5% of 252d high "
    "AND positive 63d momentum when JNK above 30d SMA AND SPY above 150d SMA; hold TLT when credit weak; "
    "rebalance every 10 bars; combines nearhi quality filter with JNK credit gate instead of 200d SMA"
)

UNIVERSE = _universe

STRATEGY = NearHiJnkCreditGated()
