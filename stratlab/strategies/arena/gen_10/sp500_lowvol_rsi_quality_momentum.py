"""SP500 low-vol enhanced RSI quality momentum.

Hypothesis (sonnet-3, gen_10):
    A two-stage selection process:
      Stage 1: Exclude SP500 stocks with RSI(14) < 40 (falling-knife / breakdown)
               AND exclude stocks in top-tercile of 63d realized volatility
               (high-vol momentum names that crash hardest in drawdowns)
      Stage 2: Rank remaining (quality, lower-vol) stocks by 126d momentum;
               hold top-15 inverse-vol weighted.

    The gen_9 RSI-quality winner (IS 0.92 OOS 0.88) used RSI floor alone.
    This strategy ALSO filters out high-vol names — it targets the intersection
    of "not breaking down" (RSI >= 40) AND "not in the high-vol cluster"
    (below 67th percentile of cross-sectional vol).

    Rationale:
      - High-vol momentum stocks have the highest beta and tend to lead market
        drawdowns disproportionately. The gen_9 RSI filter avoids broken names;
        the vol-percentile filter avoids "lottery ticket" momentum names.
      - The combined filter selects a tighter set of medium-momentum, quality
        stocks — less exciting individual names but more reliable portfolio-level
        returns.
      - Unlike `gen10_sp500_submedvol_momentum` (filters below median vol),
        this uses the 67th percentile threshold and combines with RSI.
      - Unlike `gen10_sp500_dual_quality_momentum` (RSI + 50d SMA),
        this uses RSI + vol-percentile (no SMA comparison).

    Design:
      - Compute RSI(14) and 63d realized vol for each SP500 stock.
      - Exclude stocks with RSI < 40.
      - Exclude stocks in top 33% of cross-sectional vol distribution.
      - Rank remaining by 126d momentum; hold top-15 inverse-vol weighted.
      - Portfolio vol-target: 12% annualized (21d realized), clipped 50-97%.
      - SPY 200d SMA outer gate: IEF defensive when bear.
      - Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 126      # 6-month ranking window
RSI_WINDOW = 14            # RSI filter
RSI_FLOOR = 40.0           # RSI >= 40 required
VOL_FILTER_WINDOW = 63     # vol percentile lookback
VOL_PCTILE_CEILING = 0.67  # exclude top 33% vol stocks
VOL_WINDOW = 21            # inverse-vol weight lookback
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_TARGET = 0.12
PORT_VOL_WINDOW = 21
ANNUALIZATION = 252


def _compute_rsi(prices: np.ndarray, window: int) -> float:
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


class Sp500LowvolRsiQualityMomentum(Strategy):
    """SP500 126d momentum with RSI >= 40 AND below-67th-pctile vol dual filter;
    inverse-vol weighted; portfolio vol-target (12% ann); SPY 200d outer gate to
    IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rsi_window: int = RSI_WINDOW,
        rsi_floor: float = RSI_FLOOR,
        vol_filter_window: int = VOL_FILTER_WINDOW,
        vol_pctile_ceiling: float = VOL_PCTILE_CEILING,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_min: float = EXPOSURE_MIN,
        exposure_max: float = EXPOSURE_MAX,
        vol_target: float = VOL_TARGET,
        port_vol_window: int = PORT_VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rsi_window=rsi_window,
            rsi_floor=rsi_floor,
            vol_filter_window=vol_filter_window,
            vol_pctile_ceiling=vol_pctile_ceiling,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            vol_target=vol_target,
            port_vol_window=port_vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rsi_window = int(rsi_window)
        self.rsi_floor = float(rsi_floor)
        self.vol_filter_window = int(vol_filter_window)
        self.vol_pctile_ceiling = float(vol_pctile_ceiling)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.momentum_window + self.vol_filter_window + self.port_vol_window + 10
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
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure_max
        else:
            need = self.momentum_window + self.vol_filter_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            # --- Step 1: Compute RSI and vol for all stocks ---
            raw_vols: dict[str, float] = {}
            rsi_values: dict[str, float] = {}
            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue
                # RSI
                rsi_val = _compute_rsi(col.values, self.rsi_window)
                if not np.isfinite(rsi_val):
                    continue
                # Vol (for cross-sectional percentile filter)
                tail_vf = col.values[-(self.vol_filter_window + 1):]
                if len(tail_vf) < self.vol_filter_window + 1:
                    continue
                logr_vf = np.log(tail_vf[1:] / tail_vf[:-1])
                rv_vf = float(np.std(logr_vf))
                if rv_vf <= 1e-6 or not np.isfinite(rv_vf):
                    continue
                raw_vols[sym] = rv_vf
                rsi_values[sym] = rsi_val

            if not raw_vols:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
                # Build orders and return
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

            # --- Step 2: Cross-sectional vol percentile threshold ---
            all_vols = np.array(list(raw_vols.values()))
            vol_threshold = float(np.percentile(all_vols, self.vol_pctile_ceiling * 100))

            # --- Step 3: Filter and rank ---
            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in raw_vols:
                col = prices[sym].dropna()

                # RSI filter
                if rsi_values[sym] < self.rsi_floor:
                    continue

                # Vol percentile filter (exclude top 33% by vol)
                if raw_vols[sym] > vol_threshold:
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

                # Inverse-vol weight (21d)
                tail_w = col.values[-(self.vol_window + 1):]
                if len(tail_w) < self.vol_window + 1:
                    continue
                logr_w = np.log(tail_w[1:] / tail_w[:-1])
                rv_w = float(np.std(logr_w))
                if rv_w <= 1e-6 or not np.isfinite(rv_w):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv_w

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # --- Portfolio vol-targeting ---
                port_prices = ctx.closes_window(self.port_vol_window + 5)
                port_rets = []
                n_rows = len(port_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in port_prices.columns:
                            continue
                        p_now = port_prices[sym].iloc[row_idx]
                        p_prev = port_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = self.vol_target / annual_vol if annual_vol > 1e-6 else 1.0
                    exposure = float(np.clip(scale, self.exposure_min, self.exposure_max))
                else:
                    exposure = self.exposure_max

                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

        # --- Build orders ---
        orders = []
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

NAME = "sp500_lowvol_rsi_quality_momentum"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with dual quality filter: RSI(14) >= 40 AND 63d realized vol "
    "below 67th cross-sectional percentile (exclude high-vol lottery-ticket momentum names); "
    "inverse-vol weighted; portfolio 12pct vol-target; SPY 200d SMA gate; IEF defensive; "
    "biweekly rebalance — RSI+vol-percentile filter combination is distinct from all leaderboard "
    "single-metric and dual-metric quality screens"
)

STRATEGY = Sp500LowvolRsiQualityMomentum()
