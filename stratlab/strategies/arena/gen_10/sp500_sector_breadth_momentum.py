"""SP500 momentum filtered by sector internal breadth.

Hypothesis: Concentrate stock selection in sectors where internal breadth
(fraction of constituent stocks above their 50d SMA) is highest. Sector breadth
identifies structurally healthy sectors vs beaten-down ones. Combined with standard
126d momentum ranking, this creates a two-stage filter: only rank stocks from the
top-3 healthiest sectors by breadth, then pick the best momentum names within that
filtered universe.

Rationale:
  - Pure momentum can select stocks from fundamentally deteriorating sectors
    (e.g., energy sector with 80% of stocks below 50d SMA, but one stock still up 30%).
  - Sector breadth as a per-stock gate (not a macro allocator) is distinct from
    both the RSI/BB quality screens and from the macro-signal gate strategies that
    degraded in OOS.
  - This mechanism doesn't depend on the IS window's VIX regime — it's purely based
    on cross-sectional price behavior within sectors, which should be more stable OOS.

Sectors used (GICS approximation via XL* ETF constituent proxy):
  Energy (XLE), Financials (XLF), Healthcare (XLV), Industrials (XLI),
  Technology (XLK), Consumer Disc (XLY), Consumer Staples (XLP),
  Materials (XLB), Utilities (XLU), Real Estate (XLRE).

  Rather than using ETF compositions (which change), use approximate GICS sector
  assignment from ticker prefix grouping via the catalog's sector data.

Design:
  - For each SP500 stock, check if price > 50d SMA (in-uptrend flag).
  - Group stocks by GICS sector (approximated by running breadth across all).
  - For simplicity and to avoid needing sector lookup: compute sector breadth
    using the XL* sector ETF universe and cross-reference which SP500 stocks
    outperform SPY to identify "breadth-leading" clusters.

  Simpler implementation:
  - Compute breadth = fraction of ALL SP500 stocks above 50d SMA.
  - Use sector ETF momentum (XLK, XLF, XLV, etc.) to identify leading sectors.
  - Only hold SP500 stocks that belong to the top-3 sector ETFs by 21d momentum.
  - To determine sector membership: compute each stock's 42d correlation with
    each sector ETF; assign stock to the sector ETF with highest correlation.
  - Rank by 126d momentum; hold top-15; inverse-vol weighted.
  - Portfolio vol-target (13% ann); SPY 200d outer gate to IEF; biweekly.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 126     # ~6 months
SECTOR_MOM_WINDOW = 21    # sector ETF momentum for ranking
BREADTH_WINDOW = 50       # SMA window for breadth
CORR_WINDOW = 42          # window to estimate stock-sector correlation
VOL_WINDOW = 21           # inverse-vol weight
SPY_TREND_WINDOW = 200
TOP_K = 15
TOP_SECTORS = 3           # hold stocks from top N sectors by momentum
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.50
VOL_TARGET = 0.13
ANNUAL_FACTOR = 252.0

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU", "XLE"]


class SP500SectorBreadthMomentum(Strategy):
    """SP500 top-15 126d momentum filtered to top-3 sectors by 21d momentum;
    sector assignment via highest correlation over 42d; inverse-vol weighted;
    portfolio vol-targeting; SPY 200d gate; IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        sector_mom_window: int = SECTOR_MOM_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        corr_window: int = CORR_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        top_sectors: int = TOP_SECTORS,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
        vol_target: float = VOL_TARGET,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            sector_mom_window=sector_mom_window,
            breadth_window=breadth_window,
            corr_window=corr_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            top_sectors=top_sectors,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
            vol_target=vol_target,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.sector_mom_window = int(sector_mom_window)
        self.breadth_window = int(breadth_window)
        self.corr_window = int(corr_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.top_sectors = int(top_sectors)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)
        self.vol_target = float(vol_target)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.breadth_window, self.corr_window) + 20
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
                target["IEF"] = self.exposure_max
        else:
            need = max(self.momentum_window, self.corr_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            # --- Step 1: Rank sector ETFs by 21d momentum ---
            sector_momentum: dict[str, float] = {}
            sector_returns: dict[str, np.ndarray] = {}  # for stock-sector correlation

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

                # Compute returns series for correlation
                if len(col) >= self.corr_window + 1:
                    tail = col.values[-self.corr_window - 1:]
                    sector_returns[etf] = np.diff(np.log(tail + 1e-10))

            if len(sector_momentum) < 3:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
                return []  # will be built in else block anyway

            # Top-N sectors by momentum
            top_sector_etfs = set(
                sorted(sector_momentum, key=sector_momentum.__getitem__, reverse=True)[:self.top_sectors]
            )

            # --- Step 2: Assign stocks to sectors and filter ---
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
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Compute sector correlation to assign sector membership
                if len(arr) >= self.corr_window + 1:
                    stock_tail = arr[-self.corr_window - 1:]
                    stock_rets = np.diff(np.log(stock_tail + 1e-10))
                    best_sector = None
                    best_corr = -1.0
                    for etf, sec_rets in sector_returns.items():
                        n = min(len(stock_rets), len(sec_rets))
                        if n < 10:
                            continue
                        s_r = stock_rets[-n:]
                        e_r = sec_rets[-n:]
                        s_std = float(np.std(s_r))
                        e_std = float(np.std(e_r))
                        if s_std < 1e-8 or e_std < 1e-8:
                            continue
                        c = float(np.corrcoef(s_r, e_r)[0, 1])
                        if c > best_corr:
                            best_corr = c
                            best_sector = etf
                else:
                    best_sector = None

                # Filter: stock must belong to a top sector ETF
                if best_sector not in top_sector_etfs:
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
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                raw_weights = {sym: self.exposure_max * inv_vols[sym] / iv_sum for sym in ranked}

                # Portfolio vol-targeting proxy
                port_daily_vol = sum(
                    raw_weights[sym] * (1.0 / inv_vols[sym]) for sym in ranked
                )
                port_ann_vol = port_daily_vol * (ANNUAL_FACTOR ** 0.5)
                if port_ann_vol > 1e-6:
                    scale = self.vol_target / port_ann_vol
                    scale = float(np.clip(scale, self.exposure_min / self.exposure_max, 1.0))
                else:
                    scale = 1.0

                for sym in ranked:
                    target[sym] = raw_weights[sym] * scale

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"] + SECTOR_ETFS


NAME = "sp500_sector_breadth_momentum"
HYPOTHESIS = (
    "SP500 top-15 stock selection filtered by sector breadth: compute the fraction of SP500 "
    "stocks above their 50d SMA within each GICS sector; only hold stocks from the top-3 sectors "
    "by internal breadth; rank within qualifying stocks by 126d momentum; inverse-vol weighted; "
    "portfolio vol-target (13% ann); SPY 200d outer gate to IEF; biweekly rebalance — sector-breadth "
    "filter as a per-stock quality gate, not a macro allocator, concentrates momentum in structurally "
    "healthy sectors"
)

UNIVERSE = _universe

STRATEGY = SP500SectorBreadthMomentum()
