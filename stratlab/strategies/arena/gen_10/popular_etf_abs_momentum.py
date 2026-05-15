"""Popular ETF top-5 absolute momentum with vol-targeted exposure.

Hypothesis: Simple cross-sectional momentum on a curated set of popular ETFs
(equity, bonds, commodities, sector) using 3-month return. Only hold ETFs with
positive absolute momentum. Vol-target aggregate exposure to 15% annualized vol.
SPY 200d SMA outer gate: when SPY bearish, only allow defensive ETFs (IEF, TLT).

This strategy holds ETFs not individual SP500 stocks, making it structurally
different from the dominant SP500 stock-picking cluster on the gen10 leaderboard.
The 3-month window and ETF universe create different timing than the 6-month
individual stock approaches.

Design:
  - Rank popular ETFs by 63d (3-month) return.
  - Hold top-5 ETFs with positive momentum.
  - Equal-weight within qualifying set (simpler than inv-vol for ETF rotation).
  - If no ETFs qualify: hold IEF.
  - Portfolio vol-target: scale total exposure down if realized portfolio vol
    exceeds 15% annualized.
  - SPY 200d SMA outer gate: when SPY bearish, restrict to {IEF, TLT} only.
  - Rebalance every 5 bars (weekly).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOMENTUM_WINDOW = 63      # 3-month
SPY_TREND_WINDOW = 200
TOP_K = 5
VOL_WINDOW = 21           # portfolio vol estimate
VOL_TARGET = 0.15         # 15% ann
ANNUAL_FACTOR = 252.0
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.40

# Popular ETFs covering IS (all have data back to at least 2010)
RISK_ETFS = [
    "SPY", "QQQ", "IWM", "MDY",                     # broad equity
    "XLK", "XLF", "XLV", "XLI", "XLY", "XLE",      # sector
    "EFA", "EEM",                                     # international
    "GLD", "DBC",                                     # commodities/real assets
    "VNQ",                                            # real estate
]
DEFENSIVE_ETFS = ["IEF", "TLT"]
ALL_ETFS = RISK_ETFS + DEFENSIVE_ETFS


class PopularETFAbsMomentum(Strategy):
    """Popular ETF top-5 by 63d momentum; positive absolute momentum required;
    equal-weight; portfolio vol-targeting; SPY 200d outer gate; weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        vol_target: float = VOL_TARGET,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            vol_window=vol_window,
            vol_target=vol_target,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.vol_target = float(vol_target)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.vol_window) + 10
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
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        need = self.momentum_window + self.vol_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < need - 10:
            return []

        if not spy_bull:
            # Only defensive ETFs allowed
            candidate_etfs = DEFENSIVE_ETFS
        else:
            candidate_etfs = RISK_ETFS + DEFENSIVE_ETFS

        scores: dict[str, float] = {}
        ind_vols: dict[str, float] = {}

        for sym in candidate_etfs:
            if sym not in prices.columns:
                continue
            col = prices[sym].dropna()
            if len(col) < self.momentum_window + 2:
                continue
            arr = col.values
            p_end = float(arr[-1])
            p_start = float(arr[-self.momentum_window])
            if p_start <= 0 or not np.isfinite(p_end) or not np.isfinite(p_start):
                continue
            ret = p_end / p_start - 1.0
            if ret <= 0.0:
                continue

            # Individual vol estimate
            if len(arr) >= self.vol_window + 1:
                tail = arr[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv > 1e-6:
                    ind_vols[sym] = rv
                else:
                    ind_vols[sym] = 0.01  # fallback

            scores[sym] = ret

        target: dict[str, float] = {}

        if len(scores) < 1:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure_max
        else:
            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

            # Equal-weight among selected
            base_weight = self.exposure_max / k

            # Estimate portfolio vol as average individual vol (equal-weight)
            port_vols = [ind_vols.get(s, 0.01) for s in ranked]
            port_daily_vol = float(np.mean(port_vols))
            port_ann_vol = port_daily_vol * (ANNUAL_FACTOR ** 0.5)

            if port_ann_vol > 1e-6:
                scale = self.vol_target / port_ann_vol
                scale = float(np.clip(scale, self.exposure_min / self.exposure_max, 1.0))
            else:
                scale = 1.0

            for sym in ranked:
                target[sym] = base_weight * scale

        # Build orders
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


NAME = "popular_etf_abs_momentum"
HYPOTHESIS = (
    "Popular ETF top-5 by 63d return with positive absolute momentum; equal-weight; "
    "portfolio vol-target (15% ann via 21d realized portfolio vol); SPY 200d outer gate "
    "restricts to defensive ETFs when bearish; weekly rebalance — ETF cross-section "
    "not individual stocks; distinct mechanism from all SP500 stock pickers"
)

UNIVERSE = ALL_ETFS + ["SPY"]

STRATEGY = PopularETFAbsMomentum()
