"""SP500 momentum concentrated in top-2 leading sectors.

Hypothesis: A stronger version of the sector-breadth approach. Instead of using
sector breadth (fraction above 50d SMA), use raw sector ETF 63d momentum to
identify the 2 best sectors. Then hold the top-15 SP500 stocks by 126d momentum
from ONLY those 2 sectors. More concentrated than the 3-sector version, which should
increase IS Calmar by focusing on the most momentum-driven sector stocks.

Design:
  - Rank sector ETFs (XLK, XLF, XLV, XLI, XLY, XLE, XLB, XLU, XLP) by 63d return.
  - Identify top-2 sectors.
  - Within those sectors, select SP500 stocks by highest correlation to sector ETF
    (using 42d rolling returns correlation) to assign sector membership.
  - From sector-qualified stocks: rank by 126d momentum, hold top-15.
  - Inverse-vol weighted, no portfolio vol-target (pure momentum concentration).
  - SPY 200d outer gate to IEF.
  - Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # ~6 months
SECTOR_MOM_WINDOW = 63    # sector ETF ranking window
CORR_WINDOW = 42          # stock-sector correlation assignment
VOL_WINDOW = 21           # inverse-vol weight
SPY_TREND_WINDOW = 200
TOP_K = 15
TOP_SECTORS = 2           # only top-2 sectors
EXPOSURE = 0.97

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLE", "XLB", "XLU", "XLP"]


class SP500SectorFocusMomentum(Strategy):
    """SP500 top-15 126d momentum concentrated in top-2 sector ETFs by 63d momentum;
    stock-sector assignment via 42d correlation; inverse-vol weighted; SPY 200d gate;
    IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        sector_mom_window: int = SECTOR_MOM_WINDOW,
        corr_window: int = CORR_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        top_sectors: int = TOP_SECTORS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            sector_mom_window=sector_mom_window,
            corr_window=corr_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            top_sectors=top_sectors,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.sector_mom_window = int(sector_mom_window)
        self.corr_window = int(corr_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.top_sectors = int(top_sectors)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.sector_mom_window, self.corr_window) + 10
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

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            need = max(self.momentum_window, self.sector_mom_window, self.corr_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            # Step 1: Rank sector ETFs by 63d momentum
            sector_momentum: dict[str, float] = {}
            sector_returns: dict[str, np.ndarray] = {}

            for etf in SECTOR_ETFS:
                if etf not in prices.columns:
                    continue
                col = prices[etf].dropna()
                if len(col) < self.sector_mom_window + 1:
                    continue
                p_now = float(col.iloc[-1])
                p_then = float(col.iloc[-self.sector_mom_window])
                if p_then <= 0:
                    continue
                sector_momentum[etf] = p_now / p_then - 1.0

                if len(col) >= self.corr_window + 1:
                    tail = col.values[-self.corr_window - 1:]
                    sector_returns[etf] = np.diff(np.log(tail + 1e-10))

            if len(sector_momentum) < 2:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                # Top-2 sectors
                top_sectors_set = set(
                    sorted(sector_momentum, key=sector_momentum.__getitem__, reverse=True)[:self.top_sectors]
                )

                # Step 2: Assign stocks to sectors and pick top-15 momentum stocks
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in SECTOR_ETFS or sym in ("SPY", "IEF"):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < need - 10:
                        continue

                    arr = col.values

                    # 126d momentum
                    if len(arr) < self.momentum_window + 2:
                        continue
                    p_end = float(arr[-1])
                    p_start = float(arr[-self.momentum_window])
                    if p_start <= 0 or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    # Sector assignment via 42d correlation
                    best_sector = None
                    if len(arr) >= self.corr_window + 1:
                        stock_tail = arr[-self.corr_window - 1:]
                        stock_rets = np.diff(np.log(stock_tail + 1e-10))
                        best_corr = -1.0
                        for etf, sec_rets in sector_returns.items():
                            n = min(len(stock_rets), len(sec_rets))
                            if n < 10:
                                continue
                            s_std = float(np.std(stock_rets[-n:]))
                            e_std = float(np.std(sec_rets[-n:]))
                            if s_std < 1e-8 or e_std < 1e-8:
                                continue
                            c = float(np.corrcoef(stock_rets[-n:], sec_rets[-n:])[0, 1])
                            if c > best_corr:
                                best_corr = c
                                best_sector = etf

                    # Only include stocks from top sectors
                    if best_sector not in top_sectors_set:
                        continue

                    # Inverse-vol weight
                    if len(arr) < self.vol_window + 1:
                        continue
                    tail = arr[-(self.vol_window + 1):]
                    logr = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    if "IEF" in closes_now.index:
                        target["IEF"] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
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
    return sp500_tickers() + ["IEF", "SPY"] + SECTOR_ETFS


NAME = "sp500_sector_focus_momentum"
HYPOTHESIS = (
    "SP500 top-15 126d momentum concentrated in top-2 leading sectors (by 63d sector ETF "
    "return): assign stocks to sectors via 42d rolling correlation to XL* ETFs; only rank "
    "stocks from top-2 sectors; inverse-vol weighted; no portfolio vol-target (concentration); "
    "SPY 200d outer gate to IEF; biweekly rebalance — top-2 sector focus is more concentrated "
    "than 3-sector breadth variant, different holdings than broad SP500 momentum"
)

UNIVERSE = _universe

STRATEGY = SP500SectorFocusMomentum()
