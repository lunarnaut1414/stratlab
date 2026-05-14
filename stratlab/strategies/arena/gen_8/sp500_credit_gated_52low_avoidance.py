"""SP500 Momentum with JNK Credit Gate and 52-Week-Low Avoidance — gen_8 sonnet-6

Hypothesis: Hold top-15 SP500 stocks by 126d momentum when JNK (high-yield
credit) is above its 30d MA (healthy credit) AND price is more than 1.3x
above each stock's 252d low (avoids distressed/recovering stocks). Equal-weight;
SPY 200d SMA gate; TLT defensive; biweekly rebalance.

Rationale:
- JNK credit gate: credit markets lead equity in stress — avoiding equity when
  HY credit is deteriorating reduces drawdown in credit-driven selloffs.
- 52-week-low avoidance: filtering out stocks near their annual lows removes
  distressed companies that may have momentum but high crash risk. "Price > 1.3x
  annual low" ensures we only hold stocks that have recovered and aren't bottoming.
- Together: momentum + credit health + avoidance of structural lows creates a
  quality-filtered momentum strategy unlike pure momentum or near-52w-high (which
  filters from the top).

Distinction from existing strategies:
- Different from nearhi_momentum_quality (that uses 52w HIGH proximity; this uses
  52w LOW avoidance — complementary quality filter from the other direction)
- Different from idiosyncratic_momentum (no beta computation, uses credit gate instead)
- JNK credit gate at stock-selection level combined with 52w-low avoidance is novel
- 126d window vs 63d in most existing momentum strategies
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 126   # ~6 months
LOW_WINDOW = 252        # 52-week low lookback
LOW_AVOIDANCE_MULT = 1.3  # price must be > 1.3x 252d low
JNK_MA_WINDOW = 30      # JNK 30d moving average for credit gate
TREND_WINDOW = 200      # SPY 200d SMA gate
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_JNK = "JNK"


class Sp500CreditGated52LowAvoidance(Strategy):
    """SP500 momentum with JNK credit gate and 52-week-low avoidance filter."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        low_window: int = LOW_WINDOW,
        low_avoidance_mult: float = LOW_AVOIDANCE_MULT,
        jnk_ma_window: int = JNK_MA_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            low_window=low_window,
            low_avoidance_mult=low_avoidance_mult,
            jnk_ma_window=jnk_ma_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.low_window = int(low_window)
        self.low_avoidance_mult = float(low_avoidance_mult)
        self.jnk_ma_window = int(jnk_ma_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.low_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
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
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT defensive
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Check JNK credit gate
            try:
                jnk_hist = ctx.history(_JNK)
            except KeyError:
                jnk_hist = None

            credit_healthy = True  # default to healthy if data unavailable
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma_window + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma_window:
                    jnk_now = float(jnk_close.iloc[-1])
                    jnk_ma = float(jnk_close.iloc[-self.jnk_ma_window:].mean())
                    credit_healthy = jnk_now > jnk_ma

            if not credit_healthy:
                # Credit deteriorating — hold TLT
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                # Bull + credit healthy: run stock selection
                need = max(self.low_window, self.momentum_window) + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 2:
                    return []

                scores: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _JNK):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.low_window:
                        continue

                    current_price = float(col.iloc[-1])
                    if current_price <= 0 or not np.isfinite(current_price):
                        continue

                    # 52-week low avoidance filter
                    recent_252 = col.iloc[-self.low_window:]
                    w52_low = float(recent_252.min())
                    if w52_low <= 0 or not np.isfinite(w52_low):
                        continue
                    # Must be at least 1.3x above the 52w low
                    if current_price < self.low_avoidance_mult * w52_low:
                        continue

                    # 126d momentum
                    if len(col) < self.momentum_window + 1:
                        continue
                    p_start = float(col.iloc[-self.momentum_window])
                    if p_start <= 0 or not np.isfinite(p_start):
                        continue
                    ret = current_price / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    scores[sym] = ret

                if len(scores) < 5:
                    # Not enough candidates — TLT defensive
                    if _TLT in live:
                        target[_TLT] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY, _JNK]


NAME = "sp500_credit_gated_52low_avoidance"
HYPOTHESIS = (
    "SP500 top-15 stocks by 126d momentum with JNK credit gate (JNK above 30d MA = credit healthy) "
    "AND 52-week-low avoidance filter (price > 1.3x its 252d low); equal-weight; "
    "SPY 200d SMA gate; TLT defensive; biweekly rebalance; "
    "uses credit health signal to avoid distressed companies"
)

UNIVERSE = _universe

STRATEGY = Sp500CreditGated52LowAvoidance()
