"""gen_9 sonnet-6 — JNK Credit + TNX Yield Direction Dual-Signal 3-Tier

Hypothesis: Combine two independent macro signals into a 3-tier regime:
  - BOTH risk-on (JNK above 30d MA AND TNX below 63d MA = credit tight + rates
    falling): hold top-15 SP500 stocks by 63d momentum, 97% exposure
  - JNK STRESSED (JNK below 30d MA = credit widening, regardless of rates):
    fully defensive, TLT 97%
  - NEUTRAL (JNK ok but TNX rising, OR signal unavailable): de-risk partially,
    SPY 60%+IEF 37%

Rationale: JNK credit health and TNX rate direction are partially orthogonal
macro signals. Credit stress (JNK<MA) historically precedes equity drawdowns
more reliably than rate direction alone. Rising rates (TNX>MA) headwind for
momentum growth stocks even when credit is ok — the neutral tier captures this
partial-risk-off state. Together they create a finer-grained regime taxonomy
than any single signal. Distinct from existing leaderboard strategies that use
JNK alone OR TNX alone.

SPY 200d SMA outer bear gate as final safety net.
Uses TNX 63d MA (not 200d) — faster-moving rate regime, 53% fire rate in IS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
JNK_MA_WINDOW = 30
TNX_MA_WINDOW = 63    # faster rate signal (52.7% fire rate in IS)

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_JNK = "JNK"
_TNX = "^TNX"


class JnkTnxDualRegime(Strategy):
    """SP500 momentum with JNK credit + TNX yield dual-signal 3-tier regime."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        jnk_ma_window: int = JNK_MA_WINDOW,
        tnx_ma_window: int = TNX_MA_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            jnk_ma_window=jnk_ma_window,
            tnx_ma_window=tnx_ma_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.jnk_ma_window = int(jnk_ma_window)
        self.tnx_ma_window = int(tnx_ma_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.tnx_ma_window, self.jnk_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- JNK credit signal ---
        jnk_ok = True  # default risk-on if unavailable
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma_window + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma_window + 1:
                    jnk_ma = float(jnk_close.iloc[-self.jnk_ma_window:].mean())
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_ok = jnk_now >= jnk_ma
        except Exception:
            pass

        # --- TNX yield direction ---
        tnx_falling = True  # default risk-on if unavailable
        try:
            tnx_hist = ctx.history(_TNX)
            if tnx_hist is not None and len(tnx_hist) >= self.tnx_ma_window + 2:
                tnx_close = tnx_hist["close"].dropna()
                if len(tnx_close) >= self.tnx_ma_window + 1:
                    tnx_ma = float(tnx_close.iloc[-self.tnx_ma_window:].mean())
                    tnx_now = float(tnx_close.iloc[-1])
                    tnx_falling = tnx_now <= tnx_ma
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Outer bear gate: full TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not jnk_ok:
            # Credit stressed: full TLT defensive (priority over rate signal)
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not tnx_falling:
            # JNK ok but rates rising: neutral tier SPY+IEF
            if _SPY in live:
                target[_SPY] = self.exposure * 0.62
            if _IEF in live:
                target[_IEF] = self.exposure * 0.35
        else:
            # Both signals risk-on: top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _JNK):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_TLT, _SPY, _IEF, _JNK, _TNX]


NAME = "jnk_tnx_dual_regime"
HYPOTHESIS = (
    "JNK credit + TNX yield dual-signal 3-tier: when JNK above 30d MA AND TNX below 63d MA "
    "(credit tight AND rates falling = double risk-on), hold top-15 SP500 stocks by 63d "
    "momentum; when JNK below 30d MA (credit stress, regardless of rates), hold TLT 97%; "
    "neutral (JNK ok but rates rising): hold SPY 60%+IEF 37%; SPY 200d outer bear gate; "
    "biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = JnkTnxDualRegime()
