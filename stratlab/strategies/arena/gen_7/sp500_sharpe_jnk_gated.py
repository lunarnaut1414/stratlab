"""SP500 Sharpe-momentum with JNK credit gate.

Hypothesis: rank SP500 stocks by rolling 42d Sharpe ratio (42d return / 42d
realized vol); hold top-20 when JNK is above its 20d SMA (risk-on credit);
hold TLT when credit is weak; rebalance every 10 bars.

Rationale: Pure momentum ranks by total return — stocks that climbed fast but
erratically score equally with stocks that climbed steadily. Ranking by Sharpe
ratio (return per unit of risk) over a 42d window selects stocks with both
strong returns AND low volatility during those 42 days. The JNK 20d SMA credit
gate ensures we only deploy into equity when high-yield spreads support risk
appetite. This combination is not present on the leaderboard (gen6 Sharpe
strategies failed due to correlation with other momentum strategies, but with
JNK gate it may pass).

Key distinctions:
  - Ranks by 42d Sharpe ratio (return/vol) not raw return or return-only
  - JNK 20d SMA credit gate (not VIX) for regime filtering
  - Equal-weight top-20 (similar trade count to high-frequency rebalancers)
  - 10-bar rebalance
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars
SHARPE_WINDOW = 42        # 42d return + realized vol
JNK_MA = 20              # JNK short-term SMA for credit regime
TOP_K = 20
TREND_WINDOW = 200
EXPOSURE = 0.97


class SP500SharpeJnkGated(Strategy):
    """SP500 top-20 by 42d Sharpe ratio when JNK > 20d SMA; TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        jnk_ma: int = JNK_MA,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sharpe_window=sharpe_window,
            jnk_ma=jnk_ma,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sharpe_window = int(sharpe_window)
        self.jnk_ma = int(jnk_ma)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sharpe_window + self.trend_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window + 5:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        # JNK credit gate
        risk_on = True  # default to risk-on if JNK data unavailable
        try:
            jnk_hist = ctx.history("JNK")
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_ma + 5:
                jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                jnk_last = float(jnk_close.iloc[-1])
                risk_on = jnk_last > jnk_sma
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

        if not bull or not risk_on:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Rank by Sharpe ratio over sharpe_window days
            need = self.sharpe_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.sharpe_window:
                return []

            sharpe_scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.sharpe_window + 1:
                    continue

                # Log returns over window
                tail = col.iloc[-(self.sharpe_window + 1):]
                logr = np.log(tail.values[1:] / tail.values[:-1])

                if len(logr) < self.sharpe_window:
                    continue

                mean_r = float(np.mean(logr))
                std_r = float(np.std(logr))

                if std_r <= 1e-8 or not np.isfinite(std_r) or not np.isfinite(mean_r):
                    continue

                # Daily Sharpe (mean / std of daily log returns)
                sharpe = mean_r / std_r
                if np.isfinite(sharpe):
                    sharpe_scores[sym] = sharpe

            if len(sharpe_scores) < self.top_k:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(sharpe_scores))
                ranked = sorted(sharpe_scores, key=sharpe_scores.__getitem__, reverse=True)[:k]
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


NAME = "sp500_sharpe_jnk_gated"
HYPOTHESIS = (
    "SP500 cross-sectional Sharpe momentum with JNK credit gate: rank SP500 stocks by rolling "
    "42d Sharpe ratio (return/vol), hold top-20 when JNK above 20d SMA (risk-on credit); "
    "hold TLT when credit weak; rebalance every 10 bars; Sharpe-ranked cross-section not on leaderboard"
)

UNIVERSE = _universe

STRATEGY = SP500SharpeJnkGated()
