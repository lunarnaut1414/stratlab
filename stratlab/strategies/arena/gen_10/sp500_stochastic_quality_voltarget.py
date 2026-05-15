"""SP500 momentum with Stochastic %K quality filter + portfolio vol-targeting.

Hypothesis (sonnet-8, gen_10):
    Rank SP500 stocks by 126d momentum, but only include stocks whose 14-day
    Stochastic oscillator %K >= 40. This filters out stocks in the lower half
    of their recent price range (deeply oversold on short cycle = potential
    breakdown in progress). Portfolio vol-targeting (12% ann) scales aggregate
    exposure 50-97%. SPY 200d SMA gate to IEF defensive.

Rationale:
    - Stochastic %K = (close - 14d low) / (14d high - 14d low) * 100
    - %K >= 40 means the stock is in the upper 60% of its 14-day price range —
      it has sufficient short-term upward momentum to avoid imminent breakdown.
    - This is mechanically different from RSI (momentum of price changes) and
      from 52w-high proximity (long-term quality) — stochastics capture current
      position within RECENT range, filtering short-cycle breakdown risk.
    - Combined with 126d momentum and portfolio vol-targeting (gen_9's most OOS-
      robust mechanism), this should capture trending stocks in healthy short-
      cycle positions with regime-invariant drawdown control.

Orthogonality:
    - vs RSI floor (gen9_sp500_rsi_quality_momentum): RSI = rate of change of
      price; Stochastic = position within recent price RANGE. Different filter.
    - vs near-52w-high (gen6_nearhi_momentum_quality): 14-day range (recent
      cycle) vs 252-day range (annual context). Much shorter-term view.
    - vs SMA-distance band (gen10 sonnet-4): we exclude BELOW-range stocks,
      not parabolic extensions above SMA.
    - Vol-targeting (not in any gen_10 committed idea except sonnet-2's dual
      RSI+SMA variant).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 126      # 6-month momentum
STOCH_WINDOW = 14          # 14-bar stochastic
STOCH_FLOOR = 40.0         # %K must be >= 40 (upper 60% of recent range)
VOL_WINDOW = 21            # inverse-vol weighting lookback
SPY_TREND_WINDOW = 200     # 200d SMA outer gate
TOP_K = 15
EXPOSURE = 0.97
# Vol-targeting
VOL_TARGET = 0.12          # 12% annualized portfolio vol target
PORT_VOL_WINDOW = 30       # 30d realized portfolio return vol
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


def _stochastic_k(prices: np.ndarray, window: int) -> float:
    """Compute %K = (close - window_low) / (window_high - window_low) * 100.

    Returns NaN if insufficient data or zero range.
    """
    if len(prices) < window:
        return float("nan")
    recent = prices[-window:]
    low = float(np.min(recent))
    high = float(np.max(recent))
    if (high - low) < 1e-9:
        return 50.0  # flat price = middle of range
    close = float(prices[-1])
    return (close - low) / (high - low) * 100.0


class SP500StochasticQualityVoltarget(Strategy):
    """SP500 126d momentum with Stochastic %K >= 40 quality filter; inverse-vol
    sized; portfolio vol-targeting aggregate exposure; SPY 200d gate; IEF defensive.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stoch_window: int = STOCH_WINDOW,
        stoch_floor: float = STOCH_FLOOR,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        vol_target: float = VOL_TARGET,
        port_vol_window: int = PORT_VOL_WINDOW,
        exposure_min: float = EXPOSURE_MIN,
        exposure_max: float = EXPOSURE_MAX,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stoch_window=stoch_window,
            stoch_floor=stoch_floor,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            vol_target=vol_target,
            port_vol_window=port_vol_window,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stoch_window = int(stoch_window)
        self.stoch_floor = float(stoch_floor)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.stoch_window + 10
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
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
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
                target["IEF"] = self.exposure_max
        else:
            need = self.momentum_window + self.stoch_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + self.stoch_window:
                    continue

                # Stochastic %K quality filter
                stoch_k = _stochastic_k(col.values, self.stoch_window)
                if not np.isfinite(stoch_k) or stoch_k < self.stoch_floor:
                    continue

                # 126d momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
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
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # Portfolio vol-targeting
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
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe

NAME = "sp500_stochastic_quality_voltarget"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with stochastic oscillator quality filter: only include "
    "stocks where 14d Stochastic %K >= 40 (not deeply oversold on fast momentum); "
    "inverse-vol weighted; portfolio 12pct vol-target scales aggregate exposure 50-97%; "
    "SPY 200d SMA gate to IEF; biweekly rebalance — Stochastic quality screen captures "
    "stocks in up-phase of short cycle, orthogonal to RSI/SMA/MACD quality screens on leaderboard"
)

STRATEGY = SP500StochasticQualityVoltarget()
