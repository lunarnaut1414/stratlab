"""HY-credit-gated SP500 momentum with tri-state allocation.

Hypothesis:
  Three-state allocation using JNK credit trend + SPY 200d SMA:
    State 1 (credit + equity bull): JNK above 30d SMA AND SPY above 200d SMA
                                    → top-20 SP500 stocks by 42d return
    State 2 (credit weak, equity ok): JNK below 30d SMA AND SPY above 200d SMA
                                    → JNK 50% + TLT 47% (blended credit/duration)
    State 3 (equity bear):            SPY below 200d SMA
                                    → TLT 90%

  Monthly rebalance (21 bars).

Rationale:
  The JNK signal captures credit market stress BEFORE equity markets show it.
  In State 2, rather than going to pure cash/bonds, we hold JNK + TLT because:
  - JNK provides yield support if credit turns around quickly
  - TLT provides duration protection if stress deepens

  The 42d momentum window (slightly shorter than 63d standard) captures
  more recent momentum. Monthly rebalance reduces turnover vs biweekly.

  Key difference from gen6_hy_credit_qqq_rotation (accepted):
  - This strategy holds SP500 STOCKS in the bull state (vs single ETF QQQ)
  - The credit-weak state uses JNK+TLT blend (vs pure TLT)
  - Longer rebalance (21 bars vs 5 bars)
  - Momentum-ranked SP500 in bull state → more diversified equity exposure

Diversification vs leaderboard:
  - gen6_hy_credit_qqq_rotation: similar credit gate but holds QQQ (1 ETF)
    vs this which holds 20 SP500 stocks; correlation should be moderate.
  - All pure momentum strategies: use price-only gate; this adds JNK credit.
  - gen5_vix_gated_sp500_momentum: VIX gate only; this uses JNK + SPY SMA.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

JNK_MA = 30          # JNK MA for credit signal
TREND_WINDOW = 200   # SPY 200d SMA
MOMENTUM_WINDOW = 42 # stock momentum lookback
REBALANCE_EVERY = 21 # monthly
TOP_K = 20
EXPOSURE = 0.97


class HyGatedSP500Momentum(Strategy):
    """SP500 momentum gated by JNK credit trend + SPY 200d SMA."""

    def __init__(
        self,
        jnk_ma: int = JNK_MA,
        trend_window: int = TREND_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            jnk_ma=jnk_ma,
            trend_window=trend_window,
            momentum_window=momentum_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.jnk_ma = int(jnk_ma)
        self.trend_window = int(trend_window)
        self.momentum_window = int(momentum_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 10
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

        # --- SPY 200d SMA ---
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 5:
                spy_close = spy_hist["close"].dropna()
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_bull = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # --- JNK credit trend ---
        jnk_bull = False
        try:
            jnk_hist = ctx.history("JNK")
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 1:
                jnk_close = jnk_hist["close"].dropna()
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_bull = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # State 3: equity bear — TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif spy_bull and not jnk_bull:
            # State 2: equity ok but credit weak — JNK + TLT blend
            if "JNK" in closes_now.index:
                target["JNK"] = 0.50 * self.exposure
            if "TLT" in closes_now.index:
                target["TLT"] = 0.47 * self.exposure
        else:
            # State 1: both bull — top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            etf_skip = {
                "SPY", "QQQ", "TLT", "SHY", "IEF", "GLD", "IAU", "AGG",
                "RSP", "DBC", "JNK", "LQD", "HYG", "SSO", "TQQQ",
                "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB",
                "XLRE", "XLY", "XLC",
            }
            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in etf_skip:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_start = float(col.iloc[-self.momentum_window])
                p_end = float(col.iloc[-1])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
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
    return sp500_tickers() + ["TLT", "SPY", "JNK"]


NAME = "hy_gated_sp500_momentum"
HYPOTHESIS = (
    "JNK-gated SP500 momentum tri-state: top-20 SP500 stocks by 42d return "
    "when JNK above 30d SMA AND SPY above 200d SMA; JNK 50%+TLT 47% when "
    "credit weak but equity ok; TLT 97% when SPY bear. Monthly rebalance."
)

UNIVERSE = _universe

STRATEGY = HyGatedSP500Momentum()
