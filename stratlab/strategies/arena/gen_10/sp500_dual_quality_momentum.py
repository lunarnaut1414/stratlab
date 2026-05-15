"""SP500 momentum with dual per-stock quality filter: RSI(14)>40 AND price above 50d SMA.

Hypothesis (sonnet-1, gen_10):
    Hold top-15 SP500 stocks by 126d momentum where each stock must pass BOTH:
      - RSI(14) > 40 (not oversold / falling-knife exclusion)
      - Price > 50d SMA (intermediate trend confirmed)
    Inverse-vol weighted. SPY 200d outer trend gate to IEF. Biweekly rebalance.

Rationale:
  - gen9_sp500_rsi_quality_momentum (96% OOS retention) used RSI>35 alone.
    The 50d SMA per-stock filter adds a second, faster quality gate: stocks
    above their 50d SMA are in an intermediate uptrend even if RSI is in the
    30-40 zone. The combination eliminates more "broken" momentum names.
  - RSI>40 (vs RSI>35 in gen9) is slightly tighter — better quality at cost
    of lower candidate pool. The 50d SMA filter is orthogonal: a stock can
    have RSI=42 but be below its 50d SMA (new downtrend) — excluded.
  - Both filters are per-stock structural quality checks that don't depend on
    the IS window's calm-VIX regime. They apply independently of macro state.
  - OOS retention expected: HIGH — mechanism is regime-invariant per-stock
    quality; similar to gen9 RSI-only winner at 96%.

Distinct from:
  - gen9_sp500_rsi_quality_momentum: single RSI>35 filter, no 50d SMA
  - gen7_sp500_126d_stock_50sma_goldencross: uses golden-cross (50/150 SMA)
    as market gate, not per-stock 50d SMA; different selection signal (50d vs
    200d stock SMA + golden cross composite).
  - nearhi_momentum_quality: uses price-to-52w-high proximity, not RSI.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly (~10 trading days)
MOMENTUM_WINDOW = 126   # 6-month momentum
RSI_WINDOW = 14         # standard RSI
RSI_FLOOR = 40.0        # tighter than gen9 winner (35)
STOCK_SMA_WINDOW = 50   # per-stock intermediate trend
VOL_WINDOW = 21         # for inverse-vol weights
SPY_TREND_WINDOW = 200  # outer trend gate
TOP_K = 15
EXPOSURE = 0.97


def _compute_rsi(prices: "np.ndarray", window: int) -> float:
    """Simple RSI(window) from a 1d close-price array. Returns NaN if insufficient."""
    if len(prices) < window + 1:
        return float("nan")
    deltas = np.diff(prices[-(window + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class SP500DualQualityMomentum(Strategy):
    """SP500 126d momentum with dual per-stock quality filter: RSI>40 + price above 50d SMA.

    Inverse-vol weighted. SPY 200d bear gate to IEF. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rsi_window: int = RSI_WINDOW,
        rsi_floor: float = RSI_FLOOR,
        stock_sma_window: int = STOCK_SMA_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rsi_window=rsi_window,
            rsi_floor=rsi_floor,
            stock_sma_window=stock_sma_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rsi_window = int(rsi_window)
        self.rsi_floor = float(rsi_floor)
        self.stock_sma_window = int(stock_sma_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        # Warmup: need momentum_window + rsi_window + some buffer
        warmup = self.momentum_window + self.rsi_window + 10
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
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Defensive: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Need enough history for all filters
            need = max(self.momentum_window, self.stock_sma_window) + self.rsi_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                min_bars = self.momentum_window + self.rsi_window
                if len(col) < min_bars:
                    continue

                # --- Filter 1: RSI(14) > rsi_floor ---
                rsi_val = _compute_rsi(col.values, self.rsi_window)
                if not np.isfinite(rsi_val) or rsi_val < self.rsi_floor:
                    continue

                # --- Filter 2: price above 50d SMA ---
                if len(col) < self.stock_sma_window + 2:
                    continue
                sma_50 = float(col.iloc[-self.stock_sma_window:].mean())
                current_price = float(col.iloc[-1])
                if current_price <= sma_50:
                    continue  # below intermediate trend

                # --- 126d momentum ---
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # --- Inverse-vol weight ---
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
                # Not enough quality candidates — fall back to IEF
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


NAME = "sp500_dual_quality_momentum"
HYPOTHESIS = (
    "SP500 top-15 momentum with dual per-stock quality filter: RSI(14)>40 AND price above 50d SMA; "
    "rank by 126d momentum; inverse-vol weighted; SPY 200d outer trend gate to IEF; biweekly rebalance "
    "— layered RSI+50dSMA quality screen tighter than RSI-only gen9 winner, avoids both falling-knife "
    "and below-intermediate-trend names"
)

UNIVERSE = _universe

STRATEGY = SP500DualQualityMomentum()
