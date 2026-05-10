"""SP500 risk-adjusted quality selection (Sharpe ratio ranking).

Hypothesis: Rank SP500 stocks by their 63-day Sharpe ratio (total return /
realized volatility). Select top-20 with inverse-vol position sizing.
SPY 200d SMA gate for bear markets (rotate to TLT). Biweekly rebalance.

Rationale: Pure momentum (high return) favors volatile stocks in runups.
Sharpe-ranked selection favors stocks with HIGH return PER UNIT RISK,
selecting smoother-trending, quality momentum names. This produces a
different stock list and different daily return path than equal-weighted
or even inverse-vol-weighted pure momentum strategies.

Structural distinctions:
- Selection criterion: Sharpe (return/vol) not just return
- Can select moderate-return/low-vol stocks over high-return/high-vol
- Different stock selection than nearhi_momentum or 52wk_high_breakout
- Inverse-vol weighting further emphasizes quality/stability
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SHARPE_WINDOW = 63
VOL_WINDOW = 20       # for inverse-vol sizing
TOP_K = 20
REBALANCE_EVERY = 10
TREND_WINDOW = 200
EXPOSURE = 0.97
_DEFENSIVE = {"TLT": 0.60, "IEF": 0.37}


class SP500SharpeQuality(Strategy):
    """SP500 top-20 by 63d Sharpe ratio, inverse-vol weighted, SPY-200d-gated."""

    def __init__(
        self,
        sharpe_window: int = SHARPE_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            sharpe_window=sharpe_window,
            vol_window=vol_window,
            top_k=top_k,
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.sharpe_window = int(sharpe_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.sharpe_window + 5
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

        # SPY trend gate
        bull = True
        try:
            spy_hist = ctx.history("SPY")
            spy_c = spy_hist["close"].dropna()
            if len(spy_c) >= self.trend_window:
                bull = float(spy_c.iloc[-1]) > float(spy_c.iloc[-self.trend_window:].mean())
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull:
            for sym, wt in _DEFENSIVE.items():
                if sym in live:
                    target[sym] = wt * self.exposure
        else:
            # Compute Sharpe ratio and vol for each stock
            need = self.sharpe_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.sharpe_window:
                return []

            sharpe_scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.sharpe_window:
                    continue
                # Compute daily log returns over the window
                window_prices = col.iloc[-self.sharpe_window:]
                log_rets = np.log(window_prices.values[1:] / window_prices.values[:-1])
                if len(log_rets) < 20:
                    continue
                mean_ret = float(np.mean(log_rets))
                std_ret = float(np.std(log_rets))
                if std_ret <= 1e-6 or not np.isfinite(std_ret):
                    continue
                # Annualized Sharpe (no risk-free rate adjustment)
                sharpe = mean_ret / std_ret  # proportional to annualized Sharpe
                if not np.isfinite(sharpe):
                    continue

                # Separate vol estimate for sizing (last vol_window days)
                vol_col = col.iloc[-self.vol_window - 1:]
                if len(vol_col) < self.vol_window:
                    continue
                log_rets_vol = np.log(vol_col.values[1:] / vol_col.values[:-1])
                rv = float(np.std(log_rets_vol))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                # Only include stocks with positive momentum (Sharpe > 0)
                if sharpe > 0:
                    sharpe_scores[sym] = sharpe
                    inv_vols[sym] = 1.0 / rv

            if len(sharpe_scores) < self.top_k:
                return []

            # Select top-K by Sharpe ratio
            ranked = sorted(sharpe_scores, key=sharpe_scores.__getitem__, reverse=True)
            longs = ranked[: self.top_k]

            # Inverse-vol weighted sizing
            iv_sum = sum(inv_vols[s] for s in longs)
            if iv_sum <= 0:
                return []
            for sym in longs:
                target[sym] = self.exposure * inv_vols[sym] / iv_sum

        orders: list[Order] = []

        # Sell positions not in target
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
    return sp500_tickers() + ["TLT", "IEF", "SPY"]


NAME = "sp500_sharpe_quality"
HYPOTHESIS = (
    "SP500 top-20 by 63d Sharpe ratio (return/vol) with SPY 200d SMA gate; "
    "inverse-vol weighted; TLT+IEF defensive; biweekly rebalance; "
    "selects risk-adjusted outperformers not just highest momentum"
)
UNIVERSE = _universe
STRATEGY = SP500SharpeQuality()
