"""SP500 skip-month momentum with low-volatility anomaly screen.

Hypothesis (sonnet-8, gen_10):
    Rank SP500 stocks by 126d-skip-21d momentum, but EXCLUDE stocks in the
    top-tercile of 63d realized volatility before ranking. This low-vol
    pre-filter selects from stocks that are trending with moderate volatility,
    avoiding the high-beta momentum names that generate excess drawdown in
    corrections. Inverse-vol weighted among selected. SPY 200d SMA gate to IEF.

Rationale:
    - The low-volatility anomaly (low-beta / low-vol stocks outperform on
      risk-adjusted basis) is empirically well-documented since Frazzini & Pedersen.
    - Pure momentum can concentrate in the highest-beta names during bull markets
      (tech mega-cap, biotech) which suffer the most in corrections.
    - Excluding the top-tercile of 63d realized vol from the momentum pool
      creates a "quality momentum" selection: persistent movers not lottery tickets.
    - This combines the vol-targeting-works lesson from gen_9 with a per-STOCK
      vol filter (different from portfolio-level vol-targeting).

Orthogonality:
    - vs gen9_sp500_rsi_quality_momentum: RSI floor (momentum decay signal) vs
      realized vol tercile screen (structural risk screen). Different filter.
    - vs gen9_sp500_voltarget_skipmon: portfolio-level exposure control vs per-
      stock selection filter. Both use skip-month but the selection mechanism is
      orthogonal (vol of stocks vs vol of portfolio).
    - vs gen6_nearhi_momentum_quality: near-52w-high ratio vs realized vol tercile.
    - vs inverse-vol weighting (present in many strategies): inv-vol weighting
      changes weights; low-vol FILTER changes which stocks are eligible. Different.
    - Skip-month (126d-21d) combined with low-vol tercile exclusion not on leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_LOOKBACK = 126         # 6-month momentum
MOM_SKIP = 21              # skip most recent month
VOL_FILTER_WINDOW = 63     # 63d realized vol for tercile exclusion
VOL_WEIGHT_WINDOW = 21     # shorter window for inverse-vol weighting
SPY_TREND_WINDOW = 200     # 200d SMA outer gate
TOP_K = 15
EXPOSURE = 0.97
ANNUALIZATION = 252


class SP500LowVolAnomalySkipmon(Strategy):
    """SP500 126d-skip-21d momentum with top-vol-tercile exclusion; inverse-vol
    weighted; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_filter_window: int = VOL_FILTER_WINDOW,
        vol_weight_window: int = VOL_WEIGHT_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_filter_window=vol_filter_window,
            vol_weight_window=vol_weight_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_filter_window = int(vol_filter_window)
        self.vol_weight_window = int(vol_weight_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + 10
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
                target["IEF"] = self.exposure
        else:
            need = self.mom_lookback + self.mom_skip + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_lookback + self.mom_skip:
                return []

            # First pass: compute 63d realized vol for all symbols
            realized_vols: dict[str, float] = {}
            skip_mom_rets: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue

                # Skip-month momentum: return from -(mom_lookback+skip) to -skip
                p_end = float(col.iloc[-self.mom_skip - 1])
                p_start = float(col.iloc[-(self.mom_lookback + self.mom_skip)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # 63d realized vol for tercile filter
                if len(col) < self.vol_filter_window + 1:
                    continue
                filter_tail = col.values[-(self.vol_filter_window + 1):]
                logr_filter = np.log(filter_tail[1:] / filter_tail[:-1])
                rv_filter = float(np.std(logr_filter)) * np.sqrt(ANNUALIZATION)
                if not np.isfinite(rv_filter) or rv_filter <= 1e-6:
                    continue

                # 21d inverse-vol for weighting
                if len(col) < self.vol_weight_window + 1:
                    continue
                weight_tail = col.values[-(self.vol_weight_window + 1):]
                logr_weight = np.log(weight_tail[1:] / weight_tail[:-1])
                rv_weight = float(np.std(logr_weight))
                if not np.isfinite(rv_weight) or rv_weight <= 1e-6:
                    continue

                skip_mom_rets[sym] = ret
                realized_vols[sym] = rv_filter
                inv_vols[sym] = 1.0 / rv_weight

            if len(skip_mom_rets) < 10:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                # Exclude top-tercile by realized vol (high-vol exclusion)
                vol_threshold = np.percentile(
                    list(realized_vols.values()), 66.67
                )
                eligible = {
                    sym: ret
                    for sym, ret in skip_mom_rets.items()
                    if realized_vols[sym] <= vol_threshold
                }

                if len(eligible) < 5:
                    if "IEF" in closes_now.index:
                        target["IEF"] = self.exposure
                else:
                    k = min(self.top_k, len(eligible))
                    ranked = sorted(eligible, key=eligible.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        return []
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

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

NAME = "sp500_lowvol_anomaly_skipmon"
HYPOTHESIS = (
    "SP500 top-15 by 126d-skip-21d momentum with low-vol anomaly quality screen: "
    "exclude stocks in top-tercile of 63d realized volatility (avoid high-beta momentum "
    "names that crash hardest in drawdowns); inverse-vol weighted among selected; "
    "SPY 200d SMA gate to IEF; biweekly rebalance — low-vol filter inside momentum "
    "ranking selects persistent low-volatility momentum names distinct from raw inverse-vol weighting"
)

STRATEGY = SP500LowVolAnomalySkipmon()
