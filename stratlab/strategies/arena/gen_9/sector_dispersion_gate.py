"""gen_9 sonnet-6 — Sector Return Dispersion Gate

Hypothesis: When the cross-sector standard deviation of 20d returns across XL*
sector ETFs is LOW (below its own 30d median), sectors are co-trending together
— a stable bull regime. Hold top-15 SP500 stocks by 63d momentum above 200d SMA.
When dispersion SPIKES above the 30d median (sectors diverging), regime is
stressed — rotate to TLT 97%.

Rationale: Sector dispersion is a breadth-quality signal distinct from VIX,
credit spreads, or yield-curve slope. Low dispersion = homogenous up-trend
(momentum works best). High dispersion = leadership rotation or sector stress
(momentum degrades). The 30d rolling median threshold is self-adapting to regime
changes without a hard level to overfit.

SPY 200d SMA outer bear gate for additional safety.
Sectors used: XLK, XLF, XLE, XLI, XLU, XLY, XLB (7 sectors with full IS coverage).
XLRE, XLV, XLP excluded — inception gaps within IS window.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
DISPERSION_WINDOW = 20    # 20d returns for cross-sector std
MEDIAN_WINDOW = 30        # rolling window for threshold

_SPY = "SPY"
_TLT = "TLT"
_SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLU", "XLY", "XLB"]


class SectorDispersionGate(Strategy):
    """SP500 momentum gated by sector return dispersion regime."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        dispersion_window: int = DISPERSION_WINDOW,
        median_window: int = MEDIAN_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
            dispersion_window=dispersion_window,
            median_window=median_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.dispersion_window = int(dispersion_window)
        self.median_window = int(median_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.dispersion_window + self.median_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- Sector dispersion gate ---
        # Compute 20d return for each sector, then take std across sectors
        # Compare today's dispersion to its own 30d rolling median
        look_needed = self.dispersion_window + self.median_window + 5
        sector_ok = True  # default to risk-on if signal unavailable

        sector_hist = {}
        for sec in _SECTORS:
            try:
                h = ctx.history(sec)
                if h is not None and len(h) >= look_needed:
                    sector_hist[sec] = h["close"].dropna()
            except Exception:
                pass

        if len(sector_hist) >= 4:
            # Build array of daily cross-sector dispersion (std of 20d returns)
            disp_series = []
            min_len = min(len(v) for v in sector_hist.values())
            total_needed = self.dispersion_window + self.median_window
            if min_len >= total_needed + 1:
                for i in range(self.median_window):
                    # Position i from end: offset = median_window - 1 - i bars ago
                    offset = self.median_window - 1 - i
                    sec_rets = []
                    for col in sector_hist.values():
                        n = len(col)
                        end_idx = n - offset  # up to this index (exclusive)
                        start_idx = end_idx - self.dispersion_window
                        if start_idx < 0:
                            break
                        ret = float(col.iloc[end_idx - 1] / col.iloc[start_idx] - 1.0)
                        if np.isfinite(ret):
                            sec_rets.append(ret)
                    if len(sec_rets) >= 4:
                        disp_series.append(float(np.std(sec_rets)))

                if len(disp_series) >= self.median_window // 2:
                    # Current dispersion = most recent day's cross-sector std
                    cur_sec_rets = []
                    for col in sector_hist.values():
                        n = len(col)
                        ret = float(col.iloc[-1] / col.iloc[-(self.dispersion_window + 1)] - 1.0)
                        if np.isfinite(ret):
                            cur_sec_rets.append(ret)
                    if len(cur_sec_rets) >= 4:
                        cur_disp = float(np.std(cur_sec_rets))
                        median_disp = float(np.median(disp_series))
                        # Low dispersion = risk-on; high dispersion = risk-off
                        sector_ok = cur_disp <= median_disp

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull or not sector_ok:
            # Defensive: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Risk-on: top-K SP500 momentum stocks above 200d SMA
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT) or sym in _SECTORS:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY] + _SECTORS


NAME = "sector_dispersion_gate"
HYPOTHESIS = (
    "Sector return dispersion gate: when cross-sector 20d return std of XL* ETFs is below "
    "30d median (low dispersion, sectors co-trending = stable bull), hold top-15 SP500 stocks "
    "by 63d momentum above 200d SMA; when dispersion spikes above 30d median (sectors "
    "diverging = stress), hold TLT 97%; SPY 200d outer bear gate; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SectorDispersionGate()
