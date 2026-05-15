"""SP500 momentum with dual per-stock quality filter and portfolio vol-targeting.

Hypothesis (sonnet-2, gen_10):
    Only rank SP500 stocks that pass BOTH:
      1. RSI(14) >= 40  — not in deep oversold / broken-trend territory
      2. Price >= own 200d SMA — in intermediate uptrend

    Then rank qualifying stocks by 126d momentum, hold top-15 inverse-vol
    weighted. Portfolio-level vol-targeting scales aggregate exposure to
    target 14% annualized portfolio vol (30d realized, clipped 50-97%).

    SPY 200d SMA outer gate: if SPY is below its 200d SMA, rotate to IEF.
    Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen9_sp500_rsi_quality_momentum (IS 0.92, OOS 0.88): RSI >= 35 floor only,
    no per-stock 200d SMA, no portfolio vol-targeting.
  - gen9_gen9_sp500_voltarget_skipmon (IS 1.08, OOS 0.86): vol-targeting only,
    no RSI quality screen, 126d-skip-21d (not straight 126d).
  - gen7_sp500_126d_stock_50sma_goldencross (IS 0.96, OOS 0.72): per-stock 50d
    SMA filter, no RSI, no portfolio vol-target, golden cross outer gate.
  - This strategy: RSI >= 40 (stricter than gen9_rsi) + per-stock 200d SMA (stronger
    than 50d SMA) + portfolio vol-targeting (from voltarget family) combined.
    Triple-layer quality + dynamic exposure sizing = structurally regime-invariant.

OOS resilience rationale:
  - Per-stock 200d SMA eliminates stocks in bear trends regardless of IS VIX regime.
  - RSI >= 40 eliminates stocks in near-oversold territory (accelerating falls).
  - Portfolio vol-target automatically reduces exposure in volatile regimes,
    not dependent on any VIX-level gate that becomes miscalibrated OOS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # ~6 months
RSI_WINDOW = 14
RSI_FLOOR = 40.0            # stricter than gen9's 35
STOCK_TREND_WINDOW = 200    # per-stock 200d SMA filter
VOL_WINDOW_INDIV = 21       # for inverse-vol weights
SPY_TREND_WINDOW = 200      # outer gate
TOP_K = 15
VOL_TARGET = 0.14           # 14% annualized portfolio vol target
PORT_VOL_WINDOW = 30        # 30d realized portfolio vol
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


def _compute_rsi(prices: "np.ndarray", window: int) -> float:
    """Compute RSI(window) from a 1d close-price array. Returns NaN if insufficient."""
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


class Sp500DualQualityVoltarget(Strategy):
    """SP500 126d momentum with RSI >= 40 + per-stock 200d SMA dual quality filter;
    inverse-vol weighted; portfolio vol-targeting (14% ann, 30d window, 50-97% range);
    SPY 200d outer gate to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        rsi_window: int = RSI_WINDOW,
        rsi_floor: float = RSI_FLOOR,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        vol_window_indiv: int = VOL_WINDOW_INDIV,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        vol_target: float = VOL_TARGET,
        port_vol_window: int = PORT_VOL_WINDOW,
        exposure_min: float = EXPOSURE_MIN,
        exposure_max: float = EXPOSURE_MAX,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            rsi_window=rsi_window,
            rsi_floor=rsi_floor,
            stock_trend_window=stock_trend_window,
            vol_window_indiv=vol_window_indiv,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            vol_target=vol_target,
            port_vol_window=port_vol_window,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.rsi_window = int(rsi_window)
        self.rsi_floor = float(rsi_floor)
        self.stock_trend_window = int(stock_trend_window)
        self.vol_window_indiv = int(vol_window_indiv)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.stock_trend_window + 10
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
                target["IEF"] = self.exposure_max
        else:
            # Need lookback for all indicators
            need = max(self.mom_lookback, self.stock_trend_window) + self.rsi_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + 2:
                    continue

                # Dual quality filter 1: RSI(14) >= 40
                rsi_val = _compute_rsi(col.values, self.rsi_window)
                if not np.isfinite(rsi_val) or rsi_val < self.rsi_floor:
                    continue

                # Dual quality filter 2: price above own 200d SMA
                if len(col) < self.stock_trend_window:
                    continue
                stock_sma = float(col.iloc[-self.stock_trend_window:].mean())
                stock_price = float(col.iloc[-1])
                if stock_price < stock_sma:
                    continue  # stock not in uptrend

                # 126d momentum score
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.mom_lookback])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window_indiv + 1):]
                if len(tail) < self.vol_window_indiv + 1:
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
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # --- Portfolio vol-targeting ---
                # Estimate 30d realized portfolio vol using equal-weight portfolio rets
                vol_prices = ctx.closes_window(self.port_vol_window + 5)
                port_rets = []
                n_rows = len(vol_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in vol_prices.columns:
                            continue
                        p_now = vol_prices[sym].iloc[row_idx]
                        p_prev = vol_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    if annual_vol > 1e-6:
                        scale = self.vol_target / annual_vol
                    else:
                        scale = 1.0
                    exposure = float(np.clip(scale, self.exposure_min, self.exposure_max))
                else:
                    exposure = self.exposure_max

                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

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


NAME = "sp500_dual_quality_voltarget"
HYPOTHESIS = (
    "SP500 126d momentum with dual per-stock quality filter: RSI(14) >= 40 AND price above individual "
    "200d SMA before ranking; hold top-15 inverse-vol weighted; portfolio vol-target (14% ann) scales "
    "aggregate exposure 50-97%; SPY 200d outer bear gate to IEF; biweekly rebalance — combining "
    "per-stock trend gate with RSI floor and portfolio vol-targeting for regime-invariant drawdown control"
)

UNIVERSE = _universe

STRATEGY = Sp500DualQualityVoltarget()
