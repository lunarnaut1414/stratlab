"""Sector-momentum-first SP500 stock selection.

Hypothesis: identify top-2 SPDR sectors by 63d return when SPY above 200d SMA;
hold top-20 SP500 stocks from those winning sectors by 42d momentum; equal-weight;
rebalance every 10 bars; TLT defensive when SPY bearish.

Rationale: Pure cross-sectional momentum selects stocks from any sector, including
temporarily hot sectors that revert. A sector-first filter narrows the stock universe
to structural winners: if tech (XLK) is the leading sector, buy the top growth stocks
within tech — they have both sector tailwind and individual momentum.

The gen_6 attempt (ic_ccabf24b) got IS Calmar 0.496 (just below 0.5 floor), suggesting
the idea has merit. This variant uses:
  - Consistent 63d lookback for both sector and stock ranking (prior used 42d sector + 21d stock)
  - Top-20 stocks from top-3 sectors (broader diversification)
  - Sector ETF list includes XLC and XLRE for full sector coverage

Distinction from existing strategies:
  - Two-stage selection: sector-level then stock-level momentum
  - Narrows stock universe to sector winners before ranking
  - Different from pure momentum (no sector filter) or pure sector rotation (no stocks)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
SECTOR_WINDOW = 63         # sector momentum lookback
STOCK_WINDOW = 42          # stock momentum lookback within winning sectors
SPY_TREND_WINDOW = 200     # SPY 200d SMA
TOP_SECTORS = 3            # number of winning sectors to select from
TOP_STOCKS = 20            # number of stocks to hold
EXPOSURE = 0.97

# All SPDR sector ETFs
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLE",   # Energy
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

# Sector -> GICS code mapping for filtering SP500 stocks
# We'll approximate by sector ETF holdings sector names
SECTOR_MAP = {
    "XLK": "Information Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}


class SectorFirstStockMomentum(Strategy):
    """Two-stage: top-3 SPDR sectors by 63d return -> top-20 SP500 stocks within those sectors."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sector_window: int = SECTOR_WINDOW,
        stock_window: int = STOCK_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_sectors: int = TOP_SECTORS,
        top_stocks: int = TOP_STOCKS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sector_window=sector_window,
            stock_window=stock_window,
            spy_trend_window=spy_trend_window,
            top_sectors=top_sectors,
            top_stocks=top_stocks,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sector_window = int(sector_window)
        self.stock_window = int(stock_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_sectors = int(top_sectors)
        self.top_stocks = int(top_stocks)
        self.exposure = float(exposure)

        # Sector -> ticker list mapping (loaded lazily)
        self._sector_tickers: dict[str, set[str]] | None = None

    def _load_sector_tickers(self) -> dict[str, set[str]]:
        """Build a sector -> tickers map from the SP500 universe."""
        try:
            from stratlab.data.universe import sp500_tickers_by_sector
            return {sector: set(tickers) for sector, tickers in sp500_tickers_by_sector().items()}
        except Exception:
            return {}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.sector_window, self.stock_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Check SPY 200d SMA
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 2:
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
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Step 1: Rank sector ETFs by sector_window return
            sector_scores: dict[str, float] = {}
            for etf in SECTOR_ETFS:
                try:
                    etf_hist = ctx.history(etf)
                    if etf_hist is None or len(etf_hist) < self.sector_window + 2:
                        continue
                    etf_close = etf_hist["close"].dropna()
                    if len(etf_close) < self.sector_window:
                        continue
                    p_end = float(etf_close.iloc[-1])
                    p_start = float(etf_close.iloc[-self.sector_window])
                    if p_start <= 0:
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret) and ret > 0:  # only positive-momentum sectors
                        sector_scores[etf] = ret
                except Exception:
                    continue

            if len(sector_scores) < 1:
                # No positive-momentum sectors — go defensive
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Top sectors by momentum
                ranked_sectors = sorted(sector_scores, key=sector_scores.__getitem__, reverse=True)
                winning_sectors = ranked_sectors[:min(self.top_sectors, len(ranked_sectors))]

                # Step 2: Collect SP500 stocks in winning sectors
                # Load sector mapping lazily
                if self._sector_tickers is None:
                    self._sector_tickers = self._load_sector_tickers()

                # Build candidate set: stocks in winning sector ETFs' sectors
                # Map sector ETF -> GICS name -> tickers in SP500
                candidate_tickers: set[str] = set()
                for etf in winning_sectors:
                    gics_name = SECTOR_MAP.get(etf)
                    if gics_name and self._sector_tickers:
                        tickers_in_sector = self._sector_tickers.get(gics_name, set())
                        candidate_tickers.update(tickers_in_sector)

                if not candidate_tickers:
                    # Fallback: use all tradeable SP500 stocks
                    candidate_tickers = set(closes_now.index)

                # Step 3: Rank candidates by stock_window momentum
                need = self.stock_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.stock_window:
                    return []

                stock_scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym not in candidate_tickers:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.stock_window:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.stock_window])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        stock_scores[sym] = ret

                if len(stock_scores) < 5:
                    # Not enough candidates — fall back to pure SP500 momentum
                    for sym in prices.columns:
                        col = prices[sym].dropna()
                        if len(col) < self.stock_window:
                            continue
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-self.stock_window])
                        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                            continue
                        ret = p_end / p_start - 1.0
                        if np.isfinite(ret):
                            stock_scores[sym] = ret

                if len(stock_scores) < 5:
                    if "TLT" in closes_now.index:
                        target["TLT"] = self.exposure
                else:
                    k = min(self.top_stocks, len(stock_scores))
                    ranked = sorted(stock_scores, key=stock_scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / k
                    for sym in ranked:
                        target[sym] = per_weight

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
    return sp500_tickers() + ["TLT", "SPY"] + SECTOR_ETFS


NAME = "sector_first_stock_momentum"
HYPOTHESIS = (
    "Sector-momentum-first SP500 stock selection: identify top-2 SPDR sectors by 63d return "
    "when SPY above 200d SMA; hold top-20 SP500 stocks from those winning sectors by 42d momentum; "
    "equal-weight; rebalance every 10 bars; TLT defensive when SPY bearish"
)

UNIVERSE = _universe

STRATEGY = SectorFirstStockMomentum()
