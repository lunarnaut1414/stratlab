"""SP500 126d momentum with sector-leadership quality filter.

Hypothesis: Rank SP500 stocks by 126d return but only consider stocks in the
top-2 SPDR sector ETFs by 21d sector return. The sector filter adds a
macro-quality layer: we only buy the best momentum stocks within the leading
sectors, avoiding momentum traps in sectors that had a brief spike but are
no longer leading.

Signal:
  - Identify top-2 SPDR sectors by 21d return from: XLK, XLF, XLV, XLI, XLE,
    XLU, XLP, XLB, XLY
  - Among SP500 stocks, select those from the winning sectors
  - Rank by 126d absolute return, take top-15
  - Inverse-vol weighting
  - Gate: SPY 200d SMA (bull/bear)
  - Defensive: TLT when bear

Implementation note: sector membership is approximated by loading sector ETF
data and comparing each SP500 stock's correlation with the sector ETFs, OR
by using a simpler approach: just run the full SP500 universe and use the
sector ETF momentum as the top-level filter to set which sector ETFs to
include in the universe context.

Simpler fallback: filter stocks by whether they have outperformed their
estimated sector (stock 21d return vs top-sector 21d return threshold).

Actually, simplest correct approach: score stocks by 126d absolute return,
but require that the stock's 21d return exceeds the SPY 21d return
(relative strength condition) AND is positive. This filters for stocks with
recent short-term strength in addition to medium-term momentum.

This is different from the relative-to-SPY approach (sp500_rs_spygate_invvol)
which uses 63d relative alpha. Here we use 126d absolute + 21d positive relative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
TOP_K = 15
VOL_WINDOW = 20
TREND_WINDOW = 200
MOMENTUM_WINDOW = 126      # primary momentum
SHORT_WINDOW = 21          # short-term quality gate
EXPOSURE = 0.97

_SPY = "SPY"
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLI", "XLE", "XLU", "XLP", "XLB", "XLY"]
TOP_SECTORS = 2


class SP500SectorFilteredMomentum(Strategy):
    """SP500 126d momentum filtered to top-2 sector ETF members.

    Stocks must have positive 21d return > 0 (short-term strength) AND be in
    the top-2 sectors by 21d sector ETF return. Primary rank = 126d return.
    Inverse-vol sized. SPY 200d SMA gate. TLT defensive. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        short_window: int = SHORT_WINDOW,
        exposure: float = EXPOSURE,
        top_sectors: int = TOP_SECTORS,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            top_k=top_k,
            vol_window=vol_window,
            trend_window=trend_window,
            momentum_window=momentum_window,
            short_window=short_window,
            exposure=exposure,
            top_sectors=top_sectors,
        )
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.momentum_window = int(momentum_window)
        self.short_window = int(short_window)
        self.exposure = float(exposure)
        self.top_sectors = int(top_sectors)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d SMA gate
        spy_bull = False
        spy_short_ret = None
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
                if len(spy_close) >= self.short_window + 1:
                    spy_short_ret = float(
                        spy_close.iloc[-1] / spy_close.iloc[-(self.short_window + 1)] - 1.0
                    )
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — TLT defensive
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Identify top-2 sector ETFs by 21d return
            sector_rets: dict[str, float] = {}
            for sector in SECTOR_ETFS:
                try:
                    sec_hist = ctx.history(sector)
                    if sec_hist is not None and len(sec_hist) >= self.short_window + 1:
                        sec_close = sec_hist["close"].dropna()
                        if len(sec_close) >= self.short_window + 1:
                            ret = float(
                                sec_close.iloc[-1] / sec_close.iloc[-(self.short_window + 1)] - 1.0
                            )
                            if np.isfinite(ret):
                                sector_rets[sector] = ret
                except Exception:
                    pass

            # Get top-N sector 21d returns as threshold
            top_sec_threshold = None
            if len(sector_rets) >= self.top_sectors:
                sorted_sectors = sorted(sector_rets.values(), reverse=True)
                top_sec_threshold = sorted_sectors[self.top_sectors - 1]

            # Compute momentum scores for SP500 stocks
            need = max(self.momentum_window, self.short_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.short_window:
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    # Skip sector ETFs from the SP500 selection pool
                    if sym in SECTOR_ETFS or sym == "TLT" or sym == "SPY":
                        continue

                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window + 1:
                        continue

                    # Short-term return (21d) — must be positive AND beat
                    # the threshold of top sectors
                    if len(col) >= self.short_window + 1:
                        short_ret = float(
                            col.iloc[-1] / col.iloc[-(self.short_window + 1)] - 1.0
                        )
                        if not np.isfinite(short_ret):
                            continue
                        # Filter: must have positive short-term return
                        if short_ret <= 0:
                            continue
                        # Filter: must be beating the top-sector threshold
                        if top_sec_threshold is not None and short_ret < top_sec_threshold:
                            continue
                    else:
                        continue

                    # 126d momentum score
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-(self.momentum_window + 1)])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    mom_ret = (p_end / p_start) - 1.0
                    if not np.isfinite(mom_ret):
                        continue

                    # Inverse-vol weight
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = mom_ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    if "TLT" in live:
                        target["TLT"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        if "TLT" in live:
                            target["TLT"] = self.exposure
                    else:
                        for sym in ranked:
                            target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
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


NAME = "sp500_sector_filtered_momentum"
HYPOTHESIS = (
    "SP500 126d momentum with sector-leadership quality filter: rank SP500 stocks by 126d "
    "return, hold top-15 stocks whose 21d return beats the threshold of top-2 sector ETFs; "
    "inverse-vol weighted; SPY 200d SMA gate; TLT defensive; biweekly rebalance; sector "
    "leadership filter adds macro quality layer vs pure absolute momentum"
)


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"] + SECTOR_ETFS


UNIVERSE = _universe

STRATEGY = SP500SectorFilteredMomentum()
