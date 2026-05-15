"""SP500 cross-sectional RSI-ranked momentum with 126d momentum filter.

Hypothesis (sonnet-10, gen_10):
    Rank SP500 stocks by RSI(14) descending (highest RSI = strongest recent
    upward pressure), but only consider stocks with positive 126d return (to
    stay with winners, not just bouncing stocks). Hold top-15 above SPY 200d
    SMA; inverse-vol weighted; portfolio vol-targeting at 12% ann; IEF
    defensive in bear markets.

Rationale:
  - RSI as a RANKING criterion (not a quality floor) selects a different set
    than momentum ranking: RSI captures the RATE of recent price change, not
    just the level. A stock with moderate 126d return but rapidly accelerating
    RSI gets priority over a stock with higher raw return but decelerating RSI.
  - The 126d positive-return filter acts as a mean-reversion guard (avoids
    ranking a bouncing oversold stock high purely on RSI).
  - Gen_9 winner `sp500_rsi_quality_momentum` uses RSI >= 35 as a FLOOR
    (exclude weak names). This strategy uses RSI as a RANK (select strong
    names). The selected top-15 will be a different, largely non-overlapping
    set — RSI floor keeps momentum names alive; RSI rank selects recently
    accelerating ones.
  - Portfolio vol-targeting (gen_9 lesson: structurally regime-invariant
    deleveraging) combines with per-stock RSI ranking.

Diversification angle:
  - No leaderboard strategy uses RSI as the PRIMARY RANKING criterion.
  - Distinct from RSI quality floor (gen_9), RSI oversold mean-reversion
    (gen_6 dead-end), and all raw-return momentum strategies.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_FILTER_WINDOW = 126    # 6-month positive-return filter
RSI_WINDOW = 14            # RSI ranking lookback
SPY_TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.12          # 12% annualized portfolio vol target
VOL_WINDOW = 30            # 30d realized portfolio vol lookback
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_WINDOW_INV = 21        # per-stock inverse-vol weighting
ANNUALIZATION = 252


def _compute_rsi(prices: "np.ndarray", window: int) -> float:
    """Compute RSI(window) from a 1d close-price array. Returns NaN if
    insufficient data.
    """
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


class SP500RsiRankMomentum(Strategy):
    """SP500 ranked by RSI(14) descending with 126d positive-return filter;
    portfolio vol-targeting; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOM_FILTER_WINDOW, RSI_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA gate
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
            need = MOM_FILTER_WINDOW + RSI_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOM_FILTER_WINDOW:
                return []

            # RSI scores (ranking criterion) with 126d positive-return filter
            rsi_scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                col_arr = col.values

                if len(col_arr) < MOM_FILTER_WINDOW + RSI_WINDOW + 1:
                    continue

                # 126d return filter: must be positive
                p_end = float(col_arr[-1])
                p_start = float(col_arr[-MOM_FILTER_WINDOW])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                ret_126d = p_end / p_start - 1.0
                if not np.isfinite(ret_126d) or ret_126d <= 0:
                    continue  # Skip stocks with negative 6-month return

                # RSI as ranking criterion
                rsi_val = _compute_rsi(col_arr, RSI_WINDOW)
                if not np.isfinite(rsi_val):
                    continue

                # Per-stock inverse-vol weight
                tail = col_arr[-(VOL_WINDOW_INV + 1):]
                if len(tail) < VOL_WINDOW_INV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                rsi_scores[sym] = rsi_val
                inv_vols[sym] = 1.0 / rv

            if len(rsi_scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE_MAX
            else:
                # Rank by RSI descending (highest RSI = strongest momentum)
                k = min(TOP_K, len(rsi_scores))
                ranked = sorted(rsi_scores, key=rsi_scores.__getitem__, reverse=True)[:k]

                # Portfolio vol-targeting
                vol_prices = ctx.closes_window(VOL_WINDOW + 5)
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
                        scale = VOL_TARGET / annual_vol
                    else:
                        scale = 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                # Inverse-vol weighted allocation
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

NAME = "sp500_rsi_rank_momentum"
HYPOTHESIS = (
    "SP500 top-15 ranked by RSI(14) descending (highest RSI = strongest recent upward "
    "pressure) filtered to stocks with positive 126d return; inverse-vol weighted; portfolio "
    "12pct vol-targeting (30d realized, clip 50-97%); SPY 200d outer gate to IEF; biweekly "
    "rebalance — RSI as primary RANKING criterion (not quality floor) selects recently "
    "accelerating names distinct from raw-return momentum and RSI-floor quality screens"
)

STRATEGY = SP500RsiRankMomentum()
