"""SP500 momentum with MACD-positive quality filter.

Hypothesis (sonnet-2, gen_10):
    Only rank SP500 stocks where the MACD(12,26) histogram is positive
    (i.e., 12d EMA > 26d EMA), indicating price momentum is in an active
    upswing phase — not just a residual 6-month gain from a past rally.

    Then rank qualifying stocks by 126d momentum, hold top-15 inverse-vol
    weighted. SPY 200d SMA outer gate: below SMA → IEF. Biweekly rebalance.

Diversification angle vs leaderboard:
  - gen9_sp500_rsi_quality_momentum (RSI >= 35 quality screen): different
    mechanism — RSI measures price distance from recent highs/lows; MACD
    measures current trend direction and strength via EMA crossover.
  - gen6_nearhi_momentum_quality (near-52w-high): threshold-based proximity;
    MACD is a dynamic indicator based on price change rate.
  - gen7_sp500_126d_stock_50sma_goldencross: uses price vs static SMA level;
    MACD captures momentum acceleration not just price level.
  - The MACD filter is mechanistically different: a stock can be above its SMA
    and have positive RSI but a negative MACD (decelerating momentum, recent
    short-term weakness). MACD > 0 selects only stocks actively trending up.

OOS resilience rationale:
  - MACD responds quickly to momentum shifts — stocks with MACD < 0 are showing
    short-term deceleration regardless of IS VIX regime.
  - No macro-gate signals (VIX level, yield curve, credit spreads) — pure
    per-stock price behavior filter that works in any market regime.
  - SPY 200d SMA outer gate provides market-level protection.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # ~6 months
MACD_FAST = 12              # EMA fast period
MACD_SLOW = 26              # EMA slow period
VOL_WINDOW = 21             # for inverse-vol weights
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97


def _ema(prices: "np.ndarray", period: int) -> float:
    """Compute EMA over the last `period` bars. Returns NaN if insufficient."""
    if len(prices) < period:
        return float("nan")
    alpha = 2.0 / (period + 1)
    ema_val = float(prices[-period])
    for p in prices[-period + 1:]:
        ema_val = alpha * float(p) + (1.0 - alpha) * ema_val
    return ema_val


class Sp500MacdQualityMomentum(Strategy):
    """SP500 126d momentum with MACD(12,26) histogram > 0 quality filter;
    inverse-vol weighted; SPY 200d outer gate to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        macd_fast: int = MACD_FAST,
        macd_slow: int = MACD_SLOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.macd_slow + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Need enough lookback for all indicators
            need = self.mom_lookback + self.macd_slow + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + 2:
                    continue

                # MACD quality filter: 12-EMA > 26-EMA (positive histogram)
                fast_ema = _ema(col.values, self.macd_fast)
                slow_ema = _ema(col.values, self.macd_slow)
                if not np.isfinite(fast_ema) or not np.isfinite(slow_ema):
                    continue
                if fast_ema <= slow_ema:
                    continue  # MACD histogram <= 0 — skip

                # 126d momentum score
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.mom_lookback])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Not enough MACD-positive candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # --- Build orders ---
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_macd_quality_momentum"
HYPOTHESIS = (
    "SP500 MACD-positive quality momentum: only rank SP500 stocks where MACD(12,26) histogram is "
    "positive (price momentum is accelerating/in upswing) alongside 126d momentum score; hold "
    "top-15 inverse-vol weighted; SPY 200d SMA outer gate to IEF; biweekly rebalance — MACD-positive "
    "filter avoids stocks in decelerating trends that still show 6-month gain, distinct from RSI or "
    "52w-high quality screens already on leaderboard"
)

UNIVERSE = _universe

STRATEGY = Sp500MacdQualityMomentum()
