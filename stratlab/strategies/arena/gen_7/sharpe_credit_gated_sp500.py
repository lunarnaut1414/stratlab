"""SP500 cross-sectional Sharpe ratio selection with JNK credit gate.

Hypothesis: rank SP500 stocks by 63d risk-adjusted return (return / realized vol),
hold top-20 when JNK above 30d SMA AND SPY above 200d SMA; inverse-vol weighted;
SHY+TLT 50/50 defensive; biweekly rebalance.

Rationale: pure momentum selects high-return stocks regardless of volatility.
The Sharpe ratio (return/vol) favors stocks with persistent, smooth gains over
lottery-ticket momentum names. Adding a JNK credit gate (risk-on condition)
filters out turbulent market environments where cross-sectional momentum degrades.
Inverse-vol weighting further penalizes high-volatility holdings.

Distinction from existing strategies:
  - Uses 63d Sharpe ratio (return/vol) not raw return for ranking
  - JNK 30d SMA credit gate + SPY 200d SMA trend gate (dual gate like gen6 credit strategies)
  - SHY+TLT 50/50 defensive (not pure TLT or pure SHY)
  - Different from nearhi_momentum_quality (no 52w-high filter here)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly (~10 bars)
MOMENTUM_WINDOW = 63       # ~3 months for Sharpe
VOL_WINDOW = 63            # same window for vol
JNK_MA = 30               # JNK SMA for credit regime
SPY_TREND_WINDOW = 200     # SPY SMA for market trend
TOP_K = 20
EXPOSURE = 0.97


class SharpeRatioCreditGatedSP500(Strategy):
    """SP500 top-20 by 63d Sharpe ratio; JNK credit gate + SPY 200d SMA; inverse-vol weighted."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        jnk_ma: int = JNK_MA,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            jnk_ma=jnk_ma,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.jnk_ma = int(jnk_ma)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.jnk_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # Check JNK credit regime
        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return []
        if len(jnk_hist) < self.jnk_ma + 2:
            return []
        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) < self.jnk_ma:
            return []
        jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
        jnk_risk_on = float(jnk_close.iloc[-1]) > jnk_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        risk_on = spy_bull and jnk_risk_on

        if not risk_on:
            # Defensive: SHY 50% + TLT 50%
            for sym, w in [("SHY", 0.50), ("TLT", 0.50)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: top-K by 63d Sharpe ratio, inverse-vol weighted
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            sharpe_scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 1:
                    continue

                # Log returns for the window
                tail = col.iloc[-self.momentum_window - 1:]
                logr = np.log(tail.values[1:] / tail.values[:-1])
                if len(logr) < self.momentum_window:
                    continue

                mean_ret = float(np.mean(logr))
                vol = float(np.std(logr))
                if vol <= 1e-6 or not np.isfinite(vol):
                    continue

                sharpe = mean_ret / vol  # annualization not needed for ranking
                if not np.isfinite(sharpe):
                    continue

                # Only include positive-Sharpe stocks
                if sharpe <= 0:
                    continue

                sharpe_scores[sym] = sharpe
                inv_vols[sym] = 1.0 / vol

            if len(sharpe_scores) < 5:
                # Fall back to SHY+TLT defensive
                for sym, w in [("SHY", 0.50), ("TLT", 0.50)]:
                    if sym in closes_now.index:
                        target[sym] = w * self.exposure
            else:
                k = min(self.top_k, len(sharpe_scores))
                ranked = sorted(sharpe_scores, key=sharpe_scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["TLT", "SHY", "SPY", "JNK"]


NAME = "sharpe_credit_gated_sp500"
HYPOTHESIS = (
    "SP500 cross-sectional 63d Sharpe ratio selection with JNK credit gate: rank SP500 stocks "
    "by 63d return/63d realized vol (risk-adjusted momentum), hold top-20 when JNK above 30d SMA "
    "AND SPY above 200d SMA; inverse-vol weighted; SHY+TLT 50/50 defensive; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SharpeRatioCreditGatedSP500()
