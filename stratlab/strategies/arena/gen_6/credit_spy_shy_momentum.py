"""Credit-spread-gated SP500 momentum with SHY defensive — gen_6 sonnet-2

Hypothesis:
  Use the JNK/LQD price ratio moving average as a credit regime signal.
  When JNK/LQD 20d MA > 60d MA (HY outperforming IG → spreads tightening,
  risk-on): hold top-15 SP500 stocks by 63-day momentum, equally weighted.
  When JNK/LQD 20d MA <= 60d MA (spreads widening, risk-off): hold SHY.
  Biweekly rebalance (every 10 bars).

Rationale:
  Credit spreads are a leading indicator of equity risk appetite. HY
  outperforming IG (JNK > LQD trend) signals tightening spreads and
  favorable conditions for equity momentum. The key improvement over prior
  gen_5 credit+momentum attempts:
  - Uses SHY (cash-equivalent) instead of TLT/GLD as defensive
    → no bond duration drag during 2010-2018 rising rate environment
  - Shorter 20/60d MA pair vs 20/90d in gen5 → faster regime detection
  - SP500 momentum top-15 (not top-10) → more diversification

  The credit signal provides a different timing source than VIX, SPY SMA,
  or RSP/SPY breadth → expected lower correlation to existing leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 20
SLOW_MA = 60
MOMENTUM_WINDOW = 63
TOP_K = 15
REBALANCE_EVERY = 10
EXPOSURE = 0.97
DEFENSIVE = "SHY"


class CreditSPYSHYMomentum(Strategy):
    """JNK/LQD credit-gated SP500 momentum; cash (SHY) when spreads widen."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        momentum_window: int = MOMENTUM_WINDOW,
        top_k: int = TOP_K,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            momentum_window=momentum_window,
            top_k=top_k,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.momentum_window = int(momentum_window)
        self.top_k = int(top_k)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.momentum_window) + 10
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

        # --- Credit regime: JNK/LQD ratio ---
        risk_on = True  # default to risk-on if signal unavailable
        try:
            jnk_hist = ctx.history("JNK")
            lqd_hist = ctx.history("LQD")
            if (jnk_hist is not None and len(jnk_hist) >= self.slow_ma + 5 and
                    lqd_hist is not None and len(lqd_hist) >= self.slow_ma + 5):
                jnk_close = jnk_hist["close"].dropna()
                lqd_close = lqd_hist["close"].dropna()
                # Align by index
                min_len = min(len(jnk_close), len(lqd_close))
                if min_len >= self.slow_ma:
                    jnk_arr = jnk_close.iloc[-min_len:].values
                    lqd_arr = lqd_close.iloc[-min_len:].values
                    # Compute ratio, handle zeros
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio = np.where(lqd_arr > 0, jnk_arr / lqd_arr, np.nan)
                    ratio_series = ratio[~np.isnan(ratio)]
                    if len(ratio_series) >= self.slow_ma:
                        fast = float(ratio_series[-self.fast_ma:].mean())
                        slow = float(ratio_series[-self.slow_ma:].mean())
                        risk_on = fast > slow
        except Exception:
            pass

        target: dict[str, float] = {}

        if not risk_on:
            # Risk-off: hold SHY (cash-equivalent)
            if DEFENSIVE in closes_now.index:
                target[DEFENSIVE] = self.exposure
        else:
            # Risk-on: top-K SP500 momentum stocks
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                # Not enough stocks, fall back to SHY
                if DEFENSIVE in closes_now.index:
                    target[DEFENSIVE] = self.exposure
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                longs = ranked[:self.top_k]
                per_weight = self.exposure / len(longs)
                for sym in longs:
                    target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["JNK", "LQD", "SHY"]


NAME = "credit_spy_shy_momentum"
HYPOTHESIS = (
    "Credit-spread-gated SP500 momentum with SHY defensive: when JNK/LQD 20d MA > 60d MA "
    "(spreads tightening) hold top-15 SP500 stocks by 63d momentum; when spreads widening "
    "hold SHY only (no bond duration drag); biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = CreditSPYSHYMomentum()
