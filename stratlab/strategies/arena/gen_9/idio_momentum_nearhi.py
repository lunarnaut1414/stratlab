"""Idiosyncratic momentum with near-52w-high quality gate.

Hypothesis: Combine the two highest OOS-performing signal types from prior
rounds into a single strategy:
  1. Idiosyncratic momentum (gen7 winner): rank SP500 stocks by 63d beta-adjusted
     alpha (residual return vs SPY) rather than raw return — selects stocks
     genuinely outperforming on a risk-adjusted basis.
  2. Near-52w-high quality filter (gen6 winner): further restrict candidates
     to stocks within 85% of their 252-day high — stocks still near highs show
     persistent institutional demand.

Combining both signals should produce a higher-conviction portfolio: stocks
that are both idiosyncratically strong (low-beta-adjusted outperformance) AND
qualitatively sound (near all-time-high proximity).

Design:
  - Compute beta (126d rolling vs SPY) and residual alpha for each stock.
  - Filter: only consider stocks with price >= 85% of 252d high.
  - Rank filtered stocks by idiosyncratic alpha (63d residual vs beta*SPY).
  - Hold top-15; inverse-vol weighted.
  - SPY 200d SMA outer gate — defensive: IEF.
  - Biweekly rebalance.

Distinction from leaderboard:
  - gen7_sp500_idiosyncratic_momentum: uses idio-mom but no near-hi filter.
  - gen6_nearhi_momentum_quality: uses near-hi + raw momentum, not idio-mom.
  - This strategy is the intersection: idio-mom only among near-hi stocks.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
BETA_WINDOW = 126         # regression window for beta
MOM_WINDOW = 63           # idiosyncratic momentum evaluation window
HIGH_WINDOW = 252         # 52-week high lookback
NEARHI_THRESHOLD = 0.85   # price must be >= 85% of 252d high
VOL_WINDOW = 21           # for inverse-vol weights
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97


def _compute_beta(stock_rets: "np.ndarray", spy_rets: "np.ndarray") -> float:
    """OLS beta of stock returns on SPY returns."""
    if len(stock_rets) < 10 or len(spy_rets) < 10:
        return float("nan")
    n = min(len(stock_rets), len(spy_rets))
    x = spy_rets[-n:]
    y = stock_rets[-n:]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    var_x = float(np.dot(xm, xm))
    if var_x < 1e-12:
        return float("nan")
    return float(np.dot(xm, ym) / var_x)


class IdioMomentumNearHi(Strategy):
    """Idiosyncratic momentum (beta-adjusted alpha) + near-52w-high quality gate;
    inverse-vol weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        beta_window: int = BETA_WINDOW,
        mom_window: int = MOM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            beta_window=beta_window,
            mom_window=mom_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.beta_window = int(beta_window)
        self.mom_window = int(mom_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.high_window, self.beta_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
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
            # Bear mode: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = self.high_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.high_window:
                return []

            # SPY log returns for beta calculation
            spy_col = prices.get("SPY", None)
            if spy_col is None:
                return []
            spy_col = spy_col.dropna()
            if len(spy_col) < self.beta_window + 2:
                return []
            spy_logr = np.log(spy_col.values[1:] / spy_col.values[:-1])

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue

                # Near-52w-high quality gate
                recent_252 = col.iloc[-self.high_window:]
                w52_high = float(recent_252.max())
                if w52_high <= 0 or not np.isfinite(w52_high):
                    continue
                current_price = float(col.iloc[-1])
                nearhi_ratio = current_price / w52_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue  # Below quality threshold

                # Idiosyncratic momentum (beta-adjusted alpha over mom_window)
                if len(col) < self.mom_window + 2:
                    continue
                stock_logr = np.log(col.values[1:] / col.values[:-1])

                # Align to last mom_window bars
                n = self.mom_window
                if len(stock_logr) < n or len(spy_logr) < n:
                    continue
                s_ret = stock_logr[-n:]
                m_ret = spy_logr[-n:]

                beta = _compute_beta(s_ret, m_ret)
                if not np.isfinite(beta):
                    continue

                # Cumulative idiosyncratic return: sum(stock_logr - beta * spy_logr)
                idio_ret = float(np.sum(s_ret - beta * m_ret))
                if not np.isfinite(idio_ret):
                    continue

                # Inverse-vol weight
                tail_logr = stock_logr[-(self.vol_window):]
                if len(tail_logr) < self.vol_window:
                    continue
                rv = float(np.std(tail_logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = idio_ret
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

        # Build orders
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


NAME = "idio_momentum_nearhi"
HYPOTHESIS = (
    "Idiosyncratic momentum with near-52w-high quality gate: rank SP500 stocks by 63d "
    "beta-adjusted alpha (residual vs SPY); further filter to stocks within 85% of 252d high; "
    "hold top-15 by idiosyncratic momentum passing quality gate; inverse-vol weighted; "
    "SPY 200d gate; IEF defensive; biweekly rebalance — combines best OOS signals: "
    "idio-mom from gen7 + near-hi filter from gen6"
)

UNIVERSE = _universe

STRATEGY = IdioMomentumNearHi()
