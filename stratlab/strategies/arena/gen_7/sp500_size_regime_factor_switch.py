"""SP500 factor switching based on size-regime: high-momentum vs low-vol.

Hypothesis: Use IWM/SPY 20-day relative return spread as a risk-appetite barometer.
When small-caps (IWM) outperform large-caps (SPY) over 20 days (risk-on regime):
  hold top-15 SP500 stocks by 63d momentum (momentum factor), equally weighted.
When large-caps dominate (risk-off within equities):
  hold top-15 SP500 stocks by LOWEST 21d realized volatility above 100d SMA
  (defensive factor), equally weighted.
Rotate to TLT when SPY is below its 200d SMA (bear market gate).
Rebalance every 10 bars.

Rationale: This strategy ALWAYS holds SP500 stocks in bull market — it doesn't
rotate to bonds based on risk signals within a bull market, it rotates between
WHICH TYPE of SP500 stocks to hold. In risk-on regimes (small-cap leading),
momentum stocks capture the growth wave. In risk-off regimes (large-cap leading),
low-vol stocks provide equity exposure with less drawdown. This intra-equity
factor switching is novel vs all existing strategies which either:
  - Always hold momentum stocks regardless of regime
  - Rotate to TLT/GLD/IEF when regime turns negative

Distinction from existing:
  - Intra-equity factor rotation (momentum vs low-vol), not equity vs bonds
  - IWM/SPY size regime signal is orthogonal to VIX and credit signals
  - Dual-factor pool (not single momentum rank)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # bars (~2 weeks)
RS_WINDOW = 20           # IWM vs SPY relative strength window
MOMENTUM_WINDOW = 63     # momentum factor lookback
VOL_WINDOW = 21          # low-vol factor realized vol window
STOCK_TREND_WINDOW = 100 # stock-level trend filter for low-vol stocks
TREND_WINDOW = 200       # SPY bear market gate
TOP_K = 15
EXPOSURE = 0.97


class SP500SizeRegimeFactorSwitch(Strategy):
    """Intra-equity factor switch: momentum when IWM leads SPY, low-vol when SPY leads; TLT bear."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rs_window: int = RS_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rs_window=rs_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            stock_trend_window=stock_trend_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.rs_window = int(rs_window)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.stock_trend_window = int(stock_trend_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY bear market gate
        try:
            spy_hist = ctx.history("SPY")
        except Exception:
            return []
        if spy_hist is None or len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Determine size regime: IWM vs SPY relative strength
            iwm_leading = False
            try:
                need = self.rs_window + 5
                prices_rs = ctx.closes_window(need)
                if (len(prices_rs) >= self.rs_window
                        and "IWM" in prices_rs.columns
                        and "SPY" in prices_rs.columns):
                    iwm_col = prices_rs["IWM"].dropna()
                    spy_col = prices_rs["SPY"].dropna()
                    if len(iwm_col) >= self.rs_window and len(spy_col) >= self.rs_window:
                        iwm_ret = float(iwm_col.iloc[-1] / iwm_col.iloc[-self.rs_window] - 1.0)
                        spy_ret_rs = float(spy_col.iloc[-1] / spy_col.iloc[-self.rs_window] - 1.0)
                        iwm_leading = (
                            np.isfinite(iwm_ret)
                            and np.isfinite(spy_ret_rs)
                            and iwm_ret > spy_ret_rs
                        )
            except Exception:
                pass

            need = max(self.momentum_window, self.stock_trend_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}

            if iwm_leading:
                # Risk-on: rank by 63d momentum (momentum factor)
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window + 2:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.momentum_window])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        scores[sym] = ret  # higher = better for momentum

                if not scores:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight
            else:
                # Risk-off within equity: rank by lowest 21d realized vol (low-vol factor)
                # Only include stocks above their 100d SMA
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < max(self.vol_window, self.stock_trend_window) + 2:
                        continue
                    # Stock trend filter
                    if len(col) >= self.stock_trend_window:
                        stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                        if float(col.iloc[-1]) < stock_sma:
                            continue  # Skip stocks below their own trend
                    # Realized volatility (lower = better)
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue
                    scores[sym] = -rv  # negative so higher = lower vol (better)

                if not scores:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
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
    return sp500_tickers() + ["TLT", "SPY", "IWM"]


NAME = "sp500_size_regime_factor_switch"
HYPOTHESIS = (
    "SP500 intra-equity factor switching: when IWM outperforms SPY on 20d return (risk-on) "
    "hold top-15 SP500 stocks by 63d momentum; when SPY leads (risk-off within equity) hold "
    "top-15 lowest-21d-vol SP500 stocks above 100d SMA; TLT when SPY below 200d SMA; bi-weekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SP500SizeRegimeFactorSwitch()
