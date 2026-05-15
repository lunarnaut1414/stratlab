"""SP500 momentum with per-stock ADX(14) trend-strength filter.

Hypothesis (sonnet-9, gen_10):
    ADX (Average Directional Index) measures trend strength (not direction) on
    a 0-100 scale. ADX > 25 indicates a stock is in a strong directional trend;
    ADX < 25 indicates a ranging/choppy market. By filtering to only stocks with
    ADX >= 25 before ranking by 126d momentum, we avoid chasing momentum in
    names that had a strong run but are now oscillating sideways — a classic
    failure mode of pure momentum strategies.

    Design:
      - Compute ADX(14) for each SP500 stock using Wilder smoothing.
      - Only rank stocks with ADX >= 25 (trending strongly).
      - Rank qualifying stocks by 126d momentum.
      - Hold top-15 inverse-vol weighted.
      - Portfolio vol-targeting (12% ann, 21d window) scales aggregate exposure 50-97%.
      - SPY 200d SMA outer bear gate to IEF.
      - Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen9_sp500_rsi_quality_momentum (OOS 0.88): RSI >= 35 floor — measures price
    level in oscillator form. ADX is orthogonal: measures directional STRENGTH
    of the trend, not whether the stock is oversold.
  - gen9_gen9_sp500_voltarget_skipmon (OOS 0.86): no per-stock quality screen.
  - gen7_sp500_126d_stock_50sma_goldencross: SMA-above filter, no trend-strength.
  - gen6_nearhi_momentum_quality: near-52w-high filter, not ADX directional strength.
  - No leaderboard strategy uses ADX as a quality/selection filter.

OOS resilience rationale:
  - ADX > 25 filter is regime-invariant: it fires equally in bull and bear
    markets (trending down is just as valid as trending up for ADX). The
    selection mechanism doesn't depend on the IS window's calm-VIX bias.
  - Portfolio vol-target reduces exposure mechanically in high-vol regimes
    without a VIX-level gate that could become miscalibrated OOS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # ~6 months
ADX_WINDOW = 14             # Wilder ADX period
ADX_THRESHOLD = 25.0        # minimum ADX for "trending" classification
VOL_WINDOW_INDIV = 21       # for inverse-vol weights
SPY_TREND_WINDOW = 200      # outer gate
TOP_K = 15
VOL_TARGET = 0.12           # 12% annualized portfolio vol target
PORT_VOL_WINDOW = 21        # realized portfolio vol lookback
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
    """Compute ADX(period) using Wilder smoothing.

    Returns NaN if insufficient data. ADX > 25 = trending; < 25 = ranging.
    """
    n = len(close)
    need = 2 * period + 5
    if n < need:
        return float("nan")

    # Use last (2*period+5) bars
    hi = high[-need:]
    lo = low[-need:]
    cl = close[-need:]

    # True Range
    tr = np.zeros(len(cl))
    dx_arr = np.zeros(len(cl))
    plus_dm = np.zeros(len(cl))
    minus_dm = np.zeros(len(cl))

    for i in range(1, len(cl)):
        hl = hi[i] - lo[i]
        hc = abs(hi[i] - cl[i - 1])
        lc = abs(lo[i] - cl[i - 1])
        tr[i] = max(hl, hc, lc)

        up_move = hi[i] - hi[i - 1]
        down_move = lo[i - 1] - lo[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Wilder smoothing (period bars)
    def wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        result = np.zeros(len(arr))
        result[p] = float(np.sum(arr[1 : p + 1]))
        for i in range(p + 1, len(arr)):
            result[i] = result[i - 1] - result[i - 1] / p + arr[i]
        return result

    atr = wilder_smooth(tr, period)
    plus_di_raw = wilder_smooth(plus_dm, period)
    minus_di_raw = wilder_smooth(minus_dm, period)

    # DX
    for i in range(period, len(cl)):
        if atr[i] < 1e-10:
            dx_arr[i] = 0.0
        else:
            plus_di = 100.0 * plus_di_raw[i] / atr[i]
            minus_di = 100.0 * minus_di_raw[i] / atr[i]
            denom = plus_di + minus_di
            if denom < 1e-10:
                dx_arr[i] = 0.0
            else:
                dx_arr[i] = 100.0 * abs(plus_di - minus_di) / denom

    # ADX = Wilder smooth of DX
    adx_arr = wilder_smooth(dx_arr, period)
    return float(adx_arr[-1])


class SP500AdxMomentum(Strategy):
    """SP500 126d momentum filtered by ADX(14) >= 25; inverse-vol weighted;
    portfolio vol-targeting; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + 2 * ADX_WINDOW + PORT_VOL_WINDOW + 15
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE_MAX
        else:
            need = MOM_LOOKBACK + 2 * ADX_WINDOW + 10
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + 2:
                    continue

                # 126d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_LOOKBACK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(VOL_WINDOW_INDIV + 1):]
                if len(tail) < VOL_WINDOW_INDIV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                # ADX filter — need OHLC history for each symbol
                try:
                    sym_hist = ctx.history(sym)
                except KeyError:
                    continue
                sym_hist = sym_hist.dropna(subset=["close"])
                n_bars = len(sym_hist)
                adx_need = 2 * ADX_WINDOW + 10
                if n_bars < adx_need:
                    continue

                hi_arr = sym_hist["high"].values[-adx_need:]
                lo_arr = sym_hist["low"].values[-adx_need:]
                cl_arr = sym_hist["close"].values[-adx_need:]
                adx_val = _compute_adx(hi_arr, lo_arr, cl_arr, ADX_WINDOW)
                if not np.isfinite(adx_val) or adx_val < ADX_THRESHOLD:
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE_MAX
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Portfolio vol-targeting
                vol_prices = ctx.closes_window(PORT_VOL_WINDOW + 5)
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
                    scale = VOL_TARGET / annual_vol if annual_vol > 1e-6 else 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "sp500_adx_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with per-stock ADX(14) trend-strength filter: "
    "exclude stocks with ADX below 25 (weak/choppy trend) before ranking; "
    "inverse-vol weighted; portfolio vol-target (12% ann, 21d window) scales exposure 50-97%; "
    "SPY 200d outer bear gate to IEF; biweekly rebalance — ADX screens for stocks in strong "
    "directional trends vs. ranging names, orthogonal to RSI/BB/MACD quality screens on leaderboard"
)

STRATEGY = SP500AdxMomentum()
