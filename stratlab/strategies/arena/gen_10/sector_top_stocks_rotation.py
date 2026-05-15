"""Sector-filtered SP500 momentum: top stocks from winning sectors only.

Hypothesis:
    Cross-sectional SP500 momentum improvements fail the corr gate because they all
    select from the same pool of SP500 stocks with similar quality filters. A
    structurally different approach: FIRST identify the top-3 winning sectors by 42d
    momentum (using XL* sector ETFs as sector proxies), THEN pick the top-5 SP500
    stocks from EACH winning sector for a 15-stock portfolio.

    This two-stage selection (sector then stock) is different from:
    - Flat cross-sectional ranking (all stocks compete globally)
    - Sector rotation into ETFs (no stock selection)
    - It generates within-sector-champion portfolios rather than market-wide leaders

    Design:
    - Stage 1: Rank the 11 SPDR sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP,
      XLRE, XLU, XLV, XLY) by 42d total return; select top-3 sectors
    - Stage 2: For each winning sector, rank SP500 stocks belonging to that sector
      by 63d momentum; pick top-5 from each sector
    - Weight: inverse-vol weighted across all 15 positions
    - Gate: SPY 200d SMA → IEF defensive
    - Rebalance: every 10 bars (biweekly)

    Sector assignment: use sector ETF 42d return as proxy, then we need to know which
    SP500 stocks belong to which sector. We use ctx.closes_window to rank all SP500
    stocks vs the sector ETF: stocks that have a rolling 42d correlation > 0.5 with
    the sector ETF are assigned to that sector.

    Note: The correlation-based sector assignment is an approximation — we use the
    sector ETF as a proxy for which stocks belong to it, filtering by those with
    the highest 42d correlation to each winning sector ETF.

Differentiation:
    - All gen5-10 strategies: flat cross-sectional ranking OR ETF-only rotation
    - This does TWO-STAGE ranking: sector selection then within-sector stock selection
    - The two-stage mechanism is not in any leaderboard entry
    - Within-sector stock selection produces portfolios with higher sector concentration
      but better diversification across sectors compared to flat cross-sectional
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
SECTOR_MOM_WINDOW = 42       # sector ETF momentum window
STOCK_MOM_WINDOW = 63        # within-sector stock momentum window
SECTOR_CORR_WINDOW = 42      # correlation window for sector assignment
INV_VOL_WINDOW = 21
SPY_TREND_WINDOW = 200
TOP_SECTORS = 3              # top sectors to select
STOCKS_PER_SECTOR = 5        # top stocks per sector
EXPOSURE = 0.97
MIN_SECTOR_CORR = 0.4        # min correlation for sector membership

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]


class SectorTopStocksRotation(Strategy):
    """Two-stage sector-filtered SP500 momentum.

    Stage 1: Top-3 sectors by 42d sector ETF return.
    Stage 2: Top-5 stocks per sector by 63d momentum (corr-based sector assignment).
    Inverse-vol weighted; SPY 200d gate; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sector_mom_window: int = SECTOR_MOM_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        sector_corr_window: int = SECTOR_CORR_WINDOW,
        inv_vol_window: int = INV_VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_sectors: int = TOP_SECTORS,
        stocks_per_sector: int = STOCKS_PER_SECTOR,
        exposure: float = EXPOSURE,
        min_sector_corr: float = MIN_SECTOR_CORR,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sector_mom_window=sector_mom_window,
            stock_mom_window=stock_mom_window,
            sector_corr_window=sector_corr_window,
            inv_vol_window=inv_vol_window,
            spy_trend_window=spy_trend_window,
            top_sectors=top_sectors,
            stocks_per_sector=stocks_per_sector,
            exposure=exposure,
            min_sector_corr=min_sector_corr,
        )
        self.rebalance_every = int(rebalance_every)
        self.sector_mom_window = int(sector_mom_window)
        self.stock_mom_window = int(stock_mom_window)
        self.sector_corr_window = int(sector_corr_window)
        self.inv_vol_window = int(inv_vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_sectors = int(top_sectors)
        self.stocks_per_sector = int(stocks_per_sector)
        self.exposure = float(exposure)
        self.min_sector_corr = float(min_sector_corr)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + self.sector_corr_window + self.stock_mom_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer gate
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
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # --- Stage 1: Rank sectors by 42d momentum ---
            need = max(self.sector_mom_window, self.sector_corr_window, self.stock_mom_window) + 5
            prices_wide = ctx.closes_window(need)
            if len(prices_wide) < self.sector_mom_window:
                return []

            sector_scores: dict[str, float] = {}
            sector_returns: dict[str, "np.ndarray"] = {}  # for corr-based assignment

            for etf in SECTOR_ETFS:
                if etf not in prices_wide.columns:
                    continue
                col = prices_wide[etf].dropna()
                if len(col) < self.sector_mom_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.sector_mom_window])
                if p_start <= 0:
                    continue
                sector_scores[etf] = p_end / p_start - 1.0
                # Build log-return series for correlation
                recent_col = col.values[-self.sector_corr_window:]
                if len(recent_col) >= self.sector_corr_window:
                    logr = np.diff(np.log(recent_col))
                    sector_returns[etf] = logr

            if len(sector_scores) < self.top_sectors:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                top_sector_etfs = sorted(
                    sector_scores, key=sector_scores.__getitem__, reverse=True
                )[:self.top_sectors]

                # --- Stage 2: For each winning sector, pick top-5 stocks ---
                # Assign stocks to sectors using correlation
                sector_to_stocks: dict[str, list[str]] = {etf: [] for etf in top_sector_etfs}
                stock_scores: dict[str, float] = {}
                stock_inv_vols: dict[str, float] = {}
                stock_sector: dict[str, str] = {}  # which sector the stock belongs to

                for sym in prices_wide.columns:
                    if sym in SECTOR_ETFS or sym in ("SPY", "IEF"):
                        continue
                    col = prices_wide[sym].dropna()
                    if len(col) < self.stock_mom_window + 2:
                        continue

                    p_end = float(col.iloc[-1])
                    if p_end <= 0:
                        continue

                    # Compute stock 42d log returns for corr assignment
                    recent_col = col.values[-self.sector_corr_window:]
                    if len(recent_col) < self.sector_corr_window:
                        continue
                    stock_logr = np.diff(np.log(recent_col))

                    # Find best-matching sector
                    best_sector = None
                    best_corr = self.min_sector_corr  # floor

                    for etf in top_sector_etfs:
                        if etf not in sector_returns:
                            continue
                        etf_logr = sector_returns[etf]
                        n_overlap = min(len(stock_logr), len(etf_logr))
                        if n_overlap < 15:
                            continue
                        corr = float(np.corrcoef(
                            stock_logr[-n_overlap:], etf_logr[-n_overlap:]
                        )[0, 1])
                        if np.isfinite(corr) and corr > best_corr:
                            best_corr = corr
                            best_sector = etf

                    if best_sector is None:
                        continue  # Stock doesn't correlate well with any winning sector

                    # Compute 63d momentum for ranking within sector
                    p_start = float(col.iloc[-self.stock_mom_window])
                    if p_start <= 0 or not np.isfinite(p_start):
                        continue
                    ret = p_end / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    # Inverse-vol
                    tail = col.values[-(self.inv_vol_window + 1):]
                    if len(tail) < self.inv_vol_window + 1:
                        continue
                    logr_short = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr_short))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    stock_scores[sym] = ret
                    stock_inv_vols[sym] = 1.0 / rv
                    stock_sector[sym] = best_sector

                # Select top stocks per sector
                selected: list[str] = []
                for etf in top_sector_etfs:
                    sector_stocks = [
                        s for s in stock_scores if stock_sector.get(s) == etf
                    ]
                    if not sector_stocks:
                        continue
                    sector_stocks_sorted = sorted(
                        sector_stocks, key=stock_scores.__getitem__, reverse=True
                    )[:self.stocks_per_sector]
                    selected.extend(sector_stocks_sorted)

                if len(selected) < 3:
                    if "IEF" in closes_now.index:
                        target["IEF"] = self.exposure
                else:
                    iv_sum = sum(stock_inv_vols[s] for s in selected)
                    if iv_sum <= 0:
                        return []
                    for sym in selected:
                        target[sym] = self.exposure * stock_inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
                               "XLRE", "XLU", "XLV", "XLY", "IEF", "SPY"]


NAME = "sector_top_stocks_rotation"
HYPOTHESIS = (
    "Two-stage sector-filtered SP500 momentum: Stage 1 ranks 11 SPDR sector ETFs by "
    "42d return, selects top-3 sectors; Stage 2 picks top-5 SP500 stocks from each "
    "winning sector (assigned by 42d correlation with sector ETF); 15-stock portfolio "
    "inverse-vol weighted; SPY 200d gate to IEF; biweekly rebalance — two-stage "
    "sector-then-stock selection not present in any leaderboard entry"
)

UNIVERSE = _universe

STRATEGY = SectorTopStocksRotation()
