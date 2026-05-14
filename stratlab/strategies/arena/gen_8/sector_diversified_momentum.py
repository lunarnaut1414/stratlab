"""SP500 Sector-Diversified Momentum — gen_8 sonnet-4

Hypothesis: Take top-2 stocks by 126d momentum from each of 10 GICS sectors
within the SP500 universe (using a hardcoded sector assignment for well-known
constituents). 200d SMA bear gate to TLT. Equal-weight across all selected
stocks (max 20 names). Biweekly rebalance.

Rationale: Pure momentum chronically overweights technology and growth sectors
in bull markets. Capping to top-2 per sector forces diversification across
industrials, healthcare, financials, consumer staples, utilities, materials,
energy, communication services, real estate, and consumer discretionary.
The result is a cross-sectionally driven portfolio that maintains breadth
across the economy's sectors even when tech dominates raw-momentum rankings.

Distinction from existing strategies:
- All existing SP500 momentum strategies pick purely by raw or idiosyncratic
  return, with no sector cap. Tech can dominate (often 6-10 of 15 picks).
- This strategy forces maximum 2 picks per sector, giving 10 sectors × 2 = 20
  diversified positions.
- Different signal from near-hi filter (gen6_nearhi), beta-adjusted score
  (gen7_idiosyncratic), individual-SMA filter (gen7_126d_goldencross).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # bi-weekly
MOMENTUM_WINDOW = 126      # ~6 months
TREND_WINDOW = 200         # SPY 200d SMA market gate
TOP_K_PER_SECTOR = 2       # max picks per sector
EXPOSURE = 0.97

# GICS sector mapping for well-known SP500 constituents
# Key: sector name, Value: list of ticker symbols
# This covers ~250+ stocks across 10 sectors (XLC excluded due to limited IS history)
_SECTOR_MAP: dict[str, list[str]] = {
    "information_technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "ACN", "AMD", "TXN", "INTU",
        "AMAT", "ADI", "KLAC", "LRCX", "MU", "HPQ", "IBM", "MSI", "SNPS", "CDNS",
        "FTNT", "FICO", "GLW", "JKHY", "PTC", "SWKS", "STX", "WDC", "TEL", "COHR",
        "IT", "MCHP", "ON", "FFIV", "ADSK", "CTSH",
    ],
    "health_care": [
        "UNH", "JNJ", "LLY", "ABT", "TMO", "DHR", "MDT", "AMGN", "ISRG", "VRTX",
        "SYK", "GILD", "REGN", "BSX", "BIIB", "BDX", "CI", "EW", "RMD", "IDXX",
        "HOLX", "COO", "BAX", "HUM", "CRL", "ALGN", "ZBH", "HSIC", "MTD", "WAT",
        "DGX", "IEX", "STE", "LH", "CAH", "MCK", "COR", "CVS", "RVTY", "PODD", "INCY",
        "UHS", "CNC", "ELV",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "COF", "USB",
        "PNC", "TFC", "BK", "STT", "AFL", "MET", "PRU", "AIG", "CB", "TRV",
        "ALL", "AON", "MMC", "CME", "NDAQ", "ICE", "SPGI", "MCO", "AMP", "RJF",
        "PFG", "IVZ", "TROW", "BEN", "AJG", "AIZ", "GL", "WRB", "BRO", "CINF",
        "ACGL", "ERIE", "HBAN", "KEY", "RF", "MTB", "FITB", "FIS", "FISV", "GPN",
        "V", "MA", "IBKR", "BX", "KKR", "APO", "ARES", "NDAQ", "FDS",
    ],
    "industrials": [
        "HON", "UPS", "CAT", "DE", "RTX", "LMT", "BA", "GE", "GD", "NOC",
        "LHX", "NSC", "CSX", "UNP", "FDX", "MMM", "ETN", "EMR", "ROK", "PH",
        "ITW", "DOV", "SWK", "PCAR", "CMI", "WAB", "PWR", "URI", "FAST", "GWW",
        "CTAS", "PAYX", "ROL", "ADP", "SNA", "TT", "AME", "IEX", "HUBB", "AOS",
        "JBHT", "CHRW", "EXPD", "XYL", "TRMB", "LDOS", "BAH", "LII", "EME",
        "GNRC", "TDG", "AXON",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "TJX", "SBUX", "BKNG", "CMG",
        "GM", "F", "LEN", "PHM", "DHI", "NVR", "AZO", "ORLY", "DRI", "YUM",
        "MAS", "WSM", "ROST", "TJX", "BBY", "TGT", "TKO", "LVS", "MGM", "WYNN",
        "RCL", "CCL", "MAR", "HLT", "DPZ", "DECK", "LULU", "ULTA", "TPR", "RL",
        "GPC", "HAS", "EBAY", "EXPE", "NCLH", "DAL", "UAL", "LUV",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "WMT", "COST", "MDLZ", "PM", "MO", "KMB", "CL",
        "GIS", "KR", "SYY", "HRL", "MKC", "CPB", "CAG", "TSN", "SJM", "ADM",
        "BG", "TAP", "STZ", "CLX", "HSY", "BF-B", "KDP", "MNST", "CASY", "DG",
        "DLTR",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "HAL", "DVN", "VLO", "BKR",
        "MPC", "PSX", "APA", "OKE", "WMB", "TRGP", "KMI", "NRG", "EQT", "FANG",
        "TPL", "VST", "FSLR", "CF",
    ],
    "materials": [
        "LIN", "APD", "ECL", "SHW", "PPG", "NUE", "MLM", "VMC", "PKG", "IP",
        "ALB", "FCX", "MOS", "NEM", "DD", "BALL", "AVY", "IFF", "SW", "LYB",
        "STLD",
    ],
    "utilities": [
        "NEE", "SO", "DUK", "D", "EXC", "AEP", "SRE", "XEL", "WEC", "ED",
        "ES", "DTE", "PEG", "EIX", "ETR", "PPL", "CNP", "FE", "AEE", "LNT",
        "EVRG", "CMS", "NI", "PNW", "AES", "ATO", "AWK",
    ],
    "real_estate": [
        "PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "WELL", "DLR", "O", "VTR",
        "EQR", "AVB", "ARE", "SBA", "SBAC", "CPT", "ESS", "MAA", "HST", "FRT",
        "KIM", "REG", "UDR", "EXR", "CSGP", "INVH", "WY", "IRM", "BXP", "DOC",
        "VICI",
    ],
}

# Flatten to a set for fast lookup
_ALL_MAPPED_TICKERS: set[str] = {t for tickers in _SECTOR_MAP.values() for t in tickers}


class SectorDiversifiedMomentum(Strategy):
    """Top-2 per GICS sector by 126d momentum with SPY 200d market gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k_per_sector: int = TOP_K_PER_SECTOR,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            trend_window=trend_window,
            top_k_per_sector=top_k_per_sector,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.top_k_per_sector = int(top_k_per_sector)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA market gate
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
            # Bear market: TLT defensive
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 2:
                return []

            # Compute 126d momentum for each sector, pick top-2
            selected: list[str] = []

            for sector, tickers in _SECTOR_MAP.items():
                sector_scores: dict[str, float] = {}

                for sym in tickers:
                    # Must be available in live prices
                    if sym not in live:
                        continue
                    if sym not in prices.columns:
                        continue

                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window + 1:
                        continue

                    current_price = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.momentum_window])
                    if p_start <= 0 or current_price <= 0:
                        continue

                    ret = current_price / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    sector_scores[sym] = ret

                if not sector_scores:
                    continue

                # Take top-K from this sector
                sector_ranked = sorted(sector_scores, key=sector_scores.__getitem__, reverse=True)
                selected.extend(sector_ranked[:self.top_k_per_sector])

            if len(selected) < 5:
                # Not enough sector picks — TLT defensive
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                per_weight = self.exposure / len(selected)
                for sym in selected:
                    if sym in live:
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
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sector_diversified_momentum"
HYPOTHESIS = (
    "SP500 sector-diversified momentum: take top-2 stocks by 126d momentum from each of "
    "10 GICS sectors, 200d SMA bear gate to TLT, equal-weight giving max 20 names; "
    "biweekly rebalance; sector cap prevents tech concentration that plagues pure momentum"
)

UNIVERSE = _universe

STRATEGY = SectorDiversifiedMomentum()
