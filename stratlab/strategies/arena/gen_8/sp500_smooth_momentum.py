"""SP500 Smooth-Momentum (Quality-Momentum Composite) — gen_8 sonnet-8

Hypothesis: Rank SP500 stocks by a composite score that rewards BOTH
42d return AND price smoothness (low volatility stability). Score =
0.6 * normalized_42d_return + 0.4 * vol_stability_score.

vol_stability_score = 1 - |current_21d_vol / long_run_mean_vol - 1|
(higher is better — closer to its own long-run vol level, not spiking)

Hold top-15 above 200d SMA; IEF defensive; biweekly rebalance.

Rationale: Pure momentum picks stocks in strong runs that often coincide
with vol spikes (gap-ups, earnings). Penalizing vol-elevated stocks
selects stocks with SMOOTH, sustained uptrends — fewer spike-then-reverse
patterns. This is a quality proxy using only price data (no fundamentals).

Key distinction from idiosyncratic_momentum (corr likely <0.85):
- idiosyncratic_momentum = raw_ret - beta * market_ret (market-adjusted)
- smooth_momentum = 0.6*raw_ret + 0.4*vol_stability (smooth-trend composite)
- Different filter: vol_stability removes earnings-spike winners that
  idiosyncratic_momentum would include
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
REBALANCE_EVERY = 10          # biweekly
MOM_WINDOW = 42               # momentum lookback
VOL_SHORT = 21                # current vol window
VOL_LONG = 126                # long-run vol baseline
TREND_WINDOW = 200            # SPY bear gate
TOP_K = 15
EXPOSURE = 0.97
MOM_WEIGHT = 0.6
VOL_WEIGHT = 0.4
_SPY = "SPY"
_IEF = "IEF"


class SP500SmoothMomentum(Strategy):
    """Quality-momentum composite: 42d return + vol stability score."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        vol_short: int = VOL_SHORT,
        vol_long: int = VOL_LONG,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        mom_weight: float = MOM_WEIGHT,
        vol_weight: float = VOL_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            vol_short=vol_short,
            vol_long=vol_long,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            mom_weight=mom_weight,
            vol_weight=vol_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.vol_short = int(vol_short)
        self.vol_long = int(vol_long)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.mom_weight = float(mom_weight)
        self.vol_weight = float(vol_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # ---- SPY trend gate ----
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Need window for vol calculation
            need = self.vol_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.vol_long:
                return []

            # Compute raw returns series for vol
            log_ret_df = np.log(prices / prices.shift(1)).dropna(how="all")

            # Build score for each symbol
            raw_mom_scores: list[float] = []
            raw_vol_scores: list[float] = []
            candidates: list[str] = []

            for sym in prices.columns:
                if sym in (_SPY, _IEF):
                    continue
                if sym not in live:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.vol_long + 1:
                    continue

                # 42d momentum
                mom_ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                if not np.isfinite(mom_ret):
                    continue

                # Vol stability: current 21d vol vs 126d mean vol
                if sym not in log_ret_df.columns:
                    continue
                ret_series = log_ret_df[sym].dropna()
                if len(ret_series) < self.vol_long:
                    continue

                vol_short_val = float(ret_series.iloc[-self.vol_short:].std())
                vol_long_val = float(ret_series.iloc[-self.vol_long:].std())

                if vol_long_val < 1e-8 or not np.isfinite(vol_short_val) or not np.isfinite(vol_long_val):
                    continue

                # Stability: how far current vol deviates from long-run level (lower deviation = better)
                vol_ratio_deviation = abs(vol_short_val / vol_long_val - 1.0)
                # vol_stability_score: 1 when current vol == long-run vol, lower when deviating
                vol_stability = max(0.0, 1.0 - vol_ratio_deviation)

                candidates.append(sym)
                raw_mom_scores.append(mom_ret)
                raw_vol_scores.append(vol_stability)

            if len(candidates) < 5:
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                # Cross-sectionally normalize both scores to [0,1]
                mom_arr = np.array(raw_mom_scores)
                vol_arr = np.array(raw_vol_scores)

                mom_min, mom_max = mom_arr.min(), mom_arr.max()
                if mom_max > mom_min:
                    mom_norm = (mom_arr - mom_min) / (mom_max - mom_min)
                else:
                    mom_norm = np.ones_like(mom_arr)

                vol_min, vol_max = vol_arr.min(), vol_arr.max()
                if vol_max > vol_min:
                    vol_norm = (vol_arr - vol_min) / (vol_max - vol_min)
                else:
                    vol_norm = np.ones_like(vol_arr)

                # Composite score
                composite = self.mom_weight * mom_norm + self.vol_weight * vol_norm

                # Rank and select top-K
                ranked_idx = np.argsort(composite)[::-1]
                k = min(self.top_k, len(candidates))
                selected = [candidates[i] for i in ranked_idx[:k]]

                per_weight = self.exposure / len(selected)
                for sym in selected:
                    if sym in live:
                        target[sym] = per_weight

        # ---- Build orders ----
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
    return sp500_tickers() + [_IEF, _SPY]


NAME = "sp500_smooth_momentum"
HYPOTHESIS = (
    "SP500 quality-momentum composite: rank by 0.6*norm(42d_return) + 0.4*vol_stability "
    "(vol stability = 1 - |current_21d_vol/long_126d_mean_vol - 1|); hold top-15 above "
    "200d SMA; IEF defensive; biweekly rebalance; selects stocks with smooth price action "
    "not just high returns"
)

UNIVERSE = _universe

STRATEGY = SP500SmoothMomentum()
