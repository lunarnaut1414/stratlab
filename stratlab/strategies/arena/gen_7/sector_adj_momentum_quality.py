"""Sector-adjusted momentum quality on SP500.

Hypothesis: Hold top-15 SP500 stocks ranked by 63d sector-adjusted return
(stock 63d return minus the sector ETF 63d return) with near-52w-high
quality filter (price > 80% of 52w high), inverse-vol weighted; SPY 200d
SMA gate; TLT defensive; biweekly rebalance.

Rationale:
  - Raw momentum ranking picks the sector-level winners as much as individual
    stock winners. In 2010-2018, tech stocks dominated simply because XLK
    was the strongest sector, not necessarily because individual tech stocks
    were doing something unique.
  - Sector-adjusted momentum (alpha vs sector) isolates true idiosyncratic
    outperformers — stocks beating their sector peers. These stocks show
    genuine company-specific catalysts (earnings beats, market share gains,
    pricing power) beyond sector tailwinds.
  - Combining sector-adjusted momentum with near-52w-high filter ensures we
    buy high-quality leaders, not just deep-value recoveries within sectors.
  - Inverse-vol weighting reduces concentration in high-beta idiosyncratic bets.

Distinction from existing strategies:
  - nearhi_momentum_quality uses raw 126d momentum; this uses sector-adjusted
    63d momentum (stock minus sector ETF return).
  - gen6_sp500_52wk_high_breakout uses 63d raw momentum with near-high filter;
    this removes the sector common factor.
  - The sector-adjusted signal creates a different return path even when holding
    similar stock baskets, as sector rotations won't drive rebalancing decisions.

Sector ETF mapping (GICS approximate):
  XLK → Technology stocks
  XLV → Healthcare stocks
  XLF → Financial stocks
  XLY → Consumer Discretionary stocks
  XLI → Industrial stocks
  XLE → Energy stocks
  XLU → Utilities stocks
  XLB → Materials stocks
  XLP → Consumer Staples stocks
  Other stocks → no sector adjustment (use raw momentum)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # 3-month
HIGH_WINDOW = 252          # 52-week high lookback
NEARHI_THRESHOLD = 0.80    # price must be > 80% of 52w high
VOL_WINDOW = 20            # for inverse-vol weights
TOP_K = 15
TREND_WINDOW = 200
EXPOSURE = 0.97

# Sector ETF universe for adjustment — stocks with known sector mapping
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLE", "XLU", "XLB", "XLP"]

# Approximate GICS sector assignments (major SP500 stocks)
# This is used to look up sector ETF performance for adjustment
STOCK_SECTOR_MAP = {
    # Technology (XLK)
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOG": "XLK", "GOOGL": "XLK",
    "META": "XLK", "AVGO": "XLK", "ORCL": "XLK", "CSCO": "XLK", "IBM": "XLK",
    "QCOM": "XLK", "TXN": "XLK", "ADBE": "XLK", "CRM": "XLK", "AMD": "XLK",
    "INTC": "XLK", "MU": "XLK", "KLAC": "XLK", "LRCX": "XLK", "AMAT": "XLK",
    "HPQ": "XLK", "MSI": "XLK", "GRMN": "XLK", "CTSH": "XLK", "ADP": "XLK",
    # Healthcare (XLV)
    "LLY": "XLV", "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABT": "XLV",
    "MRK": "XLV", "TMO": "XLV", "MDT": "XLV", "BMY": "XLV", "CVS": "XLV",
    "ELV": "XLV", "CI": "XLV", "HUM": "XLV", "ISRG": "XLV", "BSX": "XLV",
    "SYK": "XLV", "BDX": "XLV", "ZBH": "XLV", "BAX": "XLV", "AMGN": "XLV",
    # Financials (XLF)
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF",
    "C": "XLF", "USB": "XLF", "PNC": "XLF", "TFC": "XLF", "COF": "XLF",
    "AXP": "XLF", "BLK": "XLF", "SCHW": "XLF", "CB": "XLF", "MMC": "XLF",
    "AON": "XLF", "MET": "XLF", "PRU": "XLF", "AFL": "XLF", "PGR": "XLF",
    # Consumer Discretionary (XLY)
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY", "MCD": "XLY",
    "SBUX": "XLY", "TGT": "XLY", "LOW": "XLY", "BKNG": "XLY", "CMG": "XLY",
    "EBAY": "XLY", "DRI": "XLY", "YUM": "XLY", "MAR": "XLY", "HLT": "XLY",
    # Industrials (XLI)
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "RTX": "XLI", "GE": "XLI",
    "LMT": "XLI", "UPS": "XLI", "DE": "XLI", "MMM": "XLI", "EMR": "XLI",
    "ETN": "XLI", "ITW": "XLI", "PH": "XLI", "FDX": "XLI", "WM": "XLI",
    # Energy (XLE)
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "EOG": "XLE", "SLB": "XLE",
    "PXD": "XLE", "OXY": "XLE", "HES": "XLE", "VLO": "XLE", "MPC": "XLE",
    "PSX": "XLE", "HAL": "XLE", "KMI": "XLE", "WMB": "XLE",
    # Consumer Staples (XLP)
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "GIS": "XLP", "K": "XLP",
    "MDLZ": "XLP", "STZ": "XLP", "HSY": "XLP", "CPB": "XLP", "SJM": "XLP",
    # Materials (XLB)
    "LIN": "XLB", "APD": "XLB", "ECL": "XLB", "DD": "XLB", "NEM": "XLB",
    "FCX": "XLB", "NUE": "XLB", "VMC": "XLB", "MLM": "XLB", "PKG": "XLB",
    # Utilities (XLU)
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    "EXC": "XLU", "PCG": "XLU", "ED": "XLU", "XEL": "XLU", "PPL": "XLU",
}


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + SECTOR_ETFS + ["TLT", "SPY"]


UNIVERSE = _universe


class SectorAdjMomentumQuality(Strategy):
    """Sector-adjusted momentum quality: picks SP500 idiosyncratic outperformers."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.high_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            need = self.high_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            # Pre-compute sector ETF 63d returns for adjustment
            sector_ret: dict[str, float] = {}
            for etf in SECTOR_ETFS:
                if etf not in prices.columns:
                    continue
                col = prices[etf].dropna()
                if len(col) < self.momentum_window:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start > 0 and np.isfinite(p_start) and np.isfinite(p_end):
                    sector_ret[etf] = p_end / p_start - 1.0

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                # Skip ETFs and non-stock symbols
                if sym in SECTOR_ETFS or sym in ["TLT", "SPY", "SHY"]:
                    continue

                col = prices[sym].dropna()
                if len(col) < self.high_window:
                    continue

                # Near-52w-high quality filter
                recent_252 = col.iloc[-self.high_window:]
                w52_high = float(recent_252.max())
                if w52_high <= 0 or not np.isfinite(w52_high):
                    continue
                current_price = float(col.iloc[-1])
                nearhi_ratio = current_price / w52_high
                if nearhi_ratio < self.nearhi_threshold:
                    continue

                # 63d sector-adjusted momentum
                if len(col) < self.momentum_window + 2:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                raw_ret = p_end / p_start - 1.0
                if not np.isfinite(raw_ret):
                    continue

                # Sector adjustment
                sector_etf = STOCK_SECTOR_MAP.get(sym)
                if sector_etf and sector_etf in sector_ret:
                    adj_ret = raw_ret - sector_ret[sector_etf]
                else:
                    adj_ret = raw_ret  # no adjustment for unmapped stocks

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = adj_ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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


NAME = "sector_adj_momentum_quality"
HYPOTHESIS = (
    "SP500 sector-relative momentum quality: hold top-15 SP500 stocks by 63d return minus "
    "sector ETF 63d return (sector-adjusted idiosyncratic momentum), with price > 80% of 52w "
    "high filter, inverse-vol weighted; SPY 200d SMA gate; TLT defensive; biweekly rebalance; "
    "sector-adjusted signal removes common factor noise"
)

STRATEGY = SectorAdjMomentumQuality()
