"""SP500 63d Sharpe-Ranked Momentum with Portfolio Vol-Targeting — gen_10 sonnet-6

Hypothesis: Rank SP500 stocks by their 63d risk-adjusted return (Sharpe ratio)
rather than raw momentum. Portfolio vol-targeting at 13% provides regime-invariant
deleveraging.

Rationale:
  - Pure momentum ranks by absolute return — this doesn't penalize stocks that
    achieved their return with extreme volatility (e.g. speculative spikes).
  - Sharpe-ranked selection biases toward stocks with smooth, persistent uptrends
    (high return, low variance over 63d). These are structurally better momentum
    names with more institutional buying support.
  - Portfolio vol-targeting (proven at 80-96% OOS retention in gen9) adds regime-
    invariant deleveraging on top — the combination of quality selection signal
    AND position-sizing mechanism is both structurally robust.
  - Using 63d Sharpe (rather than 126d raw return) picks names entering new momentum
    phases vs. those fading from a prior spike.
  - Distinct from all existing strategies: no strategy has used per-stock Sharpe
    as the ranking criterion combined with portfolio vol-targeting.

Design:
  - Per-stock 63d Sharpe = (63d daily return mean / 63d daily return std) * sqrt(252).
  - Hold top-15 stocks above SPY 200d SMA.
  - Inverse-vol weighted (21d realized vol).
  - Portfolio vol-target: scale exposure = clip(13% / 30d_portfolio_rv, 50%, 97%).
  - IEF defensive when SPY below 200d SMA.
  - Biweekly rebalance (every 10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10         # biweekly
SHARPE_WINDOW = 63           # ~3 months for per-stock Sharpe
PORT_VOL_WINDOW = 30         # 30d realized portfolio vol
INV_VOL_WINDOW = 21          # per-stock inverse-vol sizing
SPY_TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.13            # 13% annualized portfolio vol target
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 0.97
SQRT252 = float(np.sqrt(252))


def _compute_stock_sharpe(prices: "np.ndarray", window: int) -> float:
    """Compute annualized Sharpe ratio over last `window` bars.

    Returns the per-stock Sharpe = (mean_daily_ret / std_daily_ret) * sqrt(252).
    Returns NaN if insufficient data or zero vol.
    """
    if len(prices) < window + 1:
        return float("nan")
    tail = prices[-(window + 1):]
    rets = np.diff(tail) / tail[:-1]
    if len(rets) < window:
        return float("nan")
    mean_ret = float(np.mean(rets))
    std_ret = float(np.std(rets))
    if std_ret <= 1e-8 or not np.isfinite(std_ret):
        return float("nan")
    return mean_ret / std_ret * SQRT252


class SP500SharpeRankVolTarget(Strategy):
    """SP500 top-15 by 63d Sharpe ratio; portfolio vol-targeted at 13%;
    inverse-vol weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sharpe_window: int = SHARPE_WINDOW,
        port_vol_window: int = PORT_VOL_WINDOW,
        inv_vol_window: int = INV_VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        vol_target: float = VOL_TARGET,
        min_exposure: float = MIN_EXPOSURE,
        max_exposure: float = MAX_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sharpe_window=sharpe_window,
            port_vol_window=port_vol_window,
            inv_vol_window=inv_vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            vol_target=vol_target,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sharpe_window = int(sharpe_window)
        self.port_vol_window = int(port_vol_window)
        self.inv_vol_window = int(inv_vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_target = float(vol_target)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)
        # Track portfolio returns for vol-targeting
        self._prev_port_value: float | None = None
        self._port_returns: list[float] = []

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.sharpe_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)

        # Track daily portfolio returns for vol-targeting
        if self._prev_port_value is not None and self._prev_port_value > 0:
            daily_ret = (equity - self._prev_port_value) / self._prev_port_value
            self._port_returns.append(daily_ret)
            if len(self._port_returns) > self.port_vol_window + 5:
                self._port_returns = self._port_returns[-(self.port_vol_window + 5):]
        self._prev_port_value = equity

        if ctx.idx % self.rebalance_every != 0:
            return []

        if equity <= 0:
            return []

        # Compute portfolio realized vol for vol-targeting
        port_vol_ann = 0.20  # default until we have history
        if len(self._port_returns) >= self.port_vol_window:
            rv = float(np.std(self._port_returns[-self.port_vol_window:]))
            if rv > 1e-8 and np.isfinite(rv):
                port_vol_ann = rv * SQRT252

        # Vol-targeted exposure
        if port_vol_ann > 1e-6:
            raw_exposure = self.vol_target / port_vol_ann
        else:
            raw_exposure = self.max_exposure
        exposure = float(np.clip(raw_exposure, self.min_exposure, self.max_exposure))

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

        target: dict[str, float] = {}

        if not spy_bull:
            # Defensive: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = exposure
        else:
            need = self.sharpe_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.sharpe_window - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.sharpe_window + 2:
                    continue

                # Per-stock 63d Sharpe ratio
                sharpe = _compute_stock_sharpe(col.values, self.sharpe_window)
                if not np.isfinite(sharpe):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.inv_vol_window + 1):]
                if len(tail) < self.inv_vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv_stock = float(np.std(logr))
                if rv_stock <= 1e-6 or not np.isfinite(rv_stock):
                    continue

                scores[sym] = sharpe
                inv_vols[sym] = 1.0 / rv_stock

            if len(scores) < 5:
                # Not enough candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

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


NAME = "sp500_sharpe_rank_voltarget"
HYPOTHESIS = (
    "SP500 cross-sectional 63d Sharpe ratio ranking with portfolio vol-targeting: rank SP500 "
    "stocks by 63d daily-return Sharpe ratio (annualized return / annualized vol); hold top-15 "
    "above SPY 200d SMA with portfolio vol-targeting at 13% (30d realized portfolio vol, clip "
    "50-97%); IEF defensive; biweekly rebalance — risk-adjusted per-stock ranking combined "
    "with portfolio vol-targeting, different mechanism from pure-return momentum ranking"
)

UNIVERSE = _universe

STRATEGY = SP500SharpeRankVolTarget()
