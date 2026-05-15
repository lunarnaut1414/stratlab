"""SP500 momentum with per-stock relative-strength filter and portfolio vol-targeting.

Hypothesis (sonnet-2, gen_10):
    Only rank SP500 stocks whose 63d return EXCEEDS SPY's 63d return
    (true alpha generators, not just market-beta participators). Then rank
    qualifying stocks by 126d momentum, hold top-15 inverse-vol weighted.

    Portfolio-level vol-targeting scales aggregate exposure to target 12%
    annualized portfolio vol (30d realized, clipped 50-97%).

    SPY 200d SMA outer gate: below SMA → IEF. Biweekly rebalance.

Diversification angle vs leaderboard:
  - gen9_sp500_rsi_quality_momentum: RSI >= 35 absolute quality screen.
  - gen9_sp500_voltarget_skipmon: vol-targeting + skip-month, no quality filter.
  - gen9_sector_slope_sp500_momentum: uses 63d alpha filter (stock beat SPY on
    same 63d window) — this strategy uses the same alpha filter but combines it
    with 126d long-horizon ranking AND portfolio vol-targeting, not a macro slope
    gate. Different mechanism and universe selection than sector_slope.
  - gen7_sp500_idiosyncratic_momentum: beta-adjusted residual alpha (stock minus
    beta*SPY), NOT raw outperformance. This strategy uses simpler 63d raw beat-
    the-index screen, which captures a different subset (high-beta stocks that
    dominate SPY still pass our filter; idiosyncratic filters them).

OOS resilience rationale:
  - Beat-the-index filter is regime-invariant: in any regime, we only hold stocks
    that are outperforming the market — natural quality signal that doesn't depend
    on VIX being calm.
  - Combining 126d long-trend ranking (who leads over 6 months) with 63d relative-
    strength screen (who's outperforming market recently) picks stocks with both
    persistent and recent leadership.
  - Portfolio vol-targeting provides the same regime-invariant deleveraging as
    gen9_sp500_voltarget_skipmon (OOS 0.86, 80% retention).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # ~6 months for ranking
REL_STRENGTH_WINDOW = 63    # 3 months for beat-the-index filter
VOL_WINDOW_INDIV = 21       # for inverse-vol weights
SPY_TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.12           # 12% annualized portfolio vol target
PORT_VOL_WINDOW = 30        # 30d realized portfolio vol
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class Sp500RelStrengthVoltarget(Strategy):
    """SP500 126d momentum with 63d beat-the-SPY relative-strength filter;
    inverse-vol weighted; portfolio vol-targeting (12% ann, 30d window, 50-97%);
    SPY 200d outer gate to IEF; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        rel_strength_window: int = REL_STRENGTH_WINDOW,
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
            rel_strength_window=rel_strength_window,
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
        self.rel_strength_window = int(rel_strength_window)
        self.vol_window_indiv = int(vol_window_indiv)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.spy_trend_window + 10
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
            need = self.mom_lookback + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            # Compute SPY's 63d return for relative-strength filter
            spy_series = spy_hist["close"].dropna()
            if len(spy_series) < self.rel_strength_window + 2:
                return []
            spy_ret_63d = (float(spy_series.iloc[-1]) / float(spy_series.iloc[-self.rel_strength_window]) - 1.0)

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + 2:
                    continue

                # Relative strength filter: 63d return must beat SPY 63d return
                if len(col) < self.rel_strength_window + 2:
                    continue
                stock_ret_63d = (float(col.iloc[-1]) / float(col.iloc[-self.rel_strength_window]) - 1.0)
                if not np.isfinite(stock_ret_63d):
                    continue
                if stock_ret_63d <= spy_ret_63d:
                    continue  # Stock not beating the market — skip

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
                # Not enough relative-strength stocks — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                # --- Portfolio vol-targeting ---
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


NAME = "sp500_relstrength_voltarget"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock relative strength filter: only include stocks whose 63d "
    "return EXCEEDS SPY 63d return (true alpha generators, not just market beta); hold top-15 "
    "inverse-vol weighted with portfolio vol-targeting (12% ann, 30d window, exposure 50-97%); "
    "SPY 200d outer gate to IEF; biweekly rebalance — relative-strength filter (beat-the-index) "
    "combined with vol-targeting is orthogonal to absolute RSI floor and near-52w-high filters on leaderboard"
)

UNIVERSE = _universe

STRATEGY = Sp500RelStrengthVoltarget()
