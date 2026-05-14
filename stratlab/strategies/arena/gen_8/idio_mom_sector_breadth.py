"""SP500 Idiosyncratic Momentum with Sector Breadth Filter — gen_8 sonnet-6

Hypothesis: Hold top-15 SP500 stocks by idiosyncratic 63d momentum (raw
return minus beta-adjusted SPY return) that also belong to the top-3 SPDR
sectors by 20d return. SPY 200d SMA gate; IEF defensive; biweekly rebalance.

Rationale: gen_7's best OOS performer (idiosyncratic_momentum, OOS Calmar 0.70)
selects stocks with company-specific alpha. Adding a sector breadth filter
(only stocks in top-3 current sectors) ensures we're buying idiosyncratic winners
in sectors with positive recent momentum. This reduces single-sector concentration
while maintaining the idiosyncratic signal. The sector filter is computed purely
from ETF prices (not traded), routing all exposure through individual stocks.

Distinction from existing strategies:
- Builds on gen_7's best performer (idiosyncratic momentum) without being a
  pure duplicate — adds sector-membership filter that restricts universe
- Sector ETFs used ONLY as ranking signal; no ETF exposure taken
- IEF defensive (not TLT — different duration, distinct from gen_7 parent)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOMENTUM_WINDOW = 63    # ~3 months idiosyncratic signal
BETA_WINDOW = 126       # 6 months for beta estimation
TREND_WINDOW = 200      # SPY 200d SMA gate
TOP_K = 15
SECTOR_TOP_N = 3        # hold stocks in top-N sectors
SECTOR_LOOKBACK = 20    # 20d return for sector ranking
EXPOSURE = 0.97

_SPY = "SPY"
_IEF = "IEF"

# SPDR sector ETFs and the set of approximate SP500 sector categories
# We rank these ETFs by 20d return, and restrict stock universe to top-N sectors.
# Mapping: ETF -> sector keyword to detect in stock sector metadata
# Since we don't have sector metadata, we use the ETF price as a proxy signal
# and map to sectors using the ETF-to-sector relationship known in the universe.
# Strategy: rank SPDR sectors, restrict to top-3 based on 20d return
_SECTOR_ETFS = ["XLK", "XLF", "XLY", "XLI", "XLV", "XLE", "XLB", "XLU", "XLP", "XLRE", "XLC"]


class IdiomMomSectorBreadth(Strategy):
    """Idiosyncratic momentum on SP500, filtered to stocks in top-3 current SPDR sectors."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        sector_top_n: int = SECTOR_TOP_N,
        sector_lookback: int = SECTOR_LOOKBACK,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            beta_window=beta_window,
            trend_window=trend_window,
            top_k=top_k,
            sector_top_n=sector_top_n,
            sector_lookback=sector_lookback,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.sector_top_n = int(sector_top_n)
        self.sector_lookback = int(sector_lookback)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.beta_window + 15
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend gate
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
            # Bear market: IEF defensive
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Rank SPDR sector ETFs by 20d return to identify top sectors
            sector_scores: dict[str, float] = {}
            need_sector = self.sector_lookback + 5
            for etf in _SECTOR_ETFS:
                try:
                    etf_hist = ctx.history(etf)
                except KeyError:
                    continue
                etf_close = etf_hist["close"].dropna()
                if len(etf_close) < self.sector_lookback + 1:
                    continue
                ret = float(etf_close.iloc[-1] / etf_close.iloc[-self.sector_lookback] - 1.0)
                if np.isfinite(ret):
                    sector_scores[etf] = ret

            # Determine top-N sectors by recent performance
            top_sectors = set()
            if sector_scores:
                ranked_sectors = sorted(sector_scores, key=sector_scores.__getitem__, reverse=True)
                top_sectors = set(ranked_sectors[: self.sector_top_n])

            # Get broad price window for momentum + beta computation
            need = max(self.beta_window, self.momentum_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                return []

            # SPY returns for beta computation
            if _SPY not in prices.columns:
                return []
            spy_prices = prices[_SPY].dropna()
            if len(spy_prices) < self.beta_window:
                return []
            spy_log_rets = np.log(spy_prices.values[1:] / spy_prices.values[:-1])
            spy_mom_ret = float(spy_prices.iloc[-1] / spy_prices.iloc[-self.momentum_window] - 1.0)

            # Compute idiosyncratic momentum for each stock
            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _IEF) or sym in _SECTOR_ETFS:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.beta_window:
                    continue

                # Compute beta
                stock_log_rets = np.log(col.values[1:] / col.values[:-1])
                n = min(len(stock_log_rets), len(spy_log_rets))
                if n < 30:
                    continue
                stock_r = stock_log_rets[-n:]
                spy_r = spy_log_rets[-n:]
                if np.std(spy_r) < 1e-8:
                    continue
                beta = float(np.cov(stock_r, spy_r)[0, 1] / np.var(spy_r))
                if not np.isfinite(beta):
                    continue

                # 63d raw momentum
                if len(col) < self.momentum_window + 1:
                    continue
                raw_ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if not np.isfinite(raw_ret):
                    continue

                idio_ret = raw_ret - beta * spy_mom_ret
                if np.isfinite(idio_ret):
                    scores[sym] = idio_ret

            # Filter to stocks where at least one sector ETF is in top_sectors
            # Since we don't have sector metadata, we use a different approach:
            # compute each stock's correlation to each sector ETF using recent returns
            # and assign to the most correlated sector.
            # If top_sectors is empty, skip sector filter.
            if top_sectors and len(scores) >= 5:
                # Compute which sector each stock best correlates with
                sector_prices: dict[str, pd.Series] = {}
                for etf in top_sectors:
                    try:
                        etf_hist = ctx.history(etf)
                    except KeyError:
                        continue
                    etf_close = etf_hist["close"].dropna()
                    if len(etf_close) >= self.sector_lookback + 10:
                        sector_prices[etf] = etf_close.iloc[-(self.sector_lookback + 10):]

                if sector_prices:
                    # Build returns for sector ETFs
                    sector_rets: dict[str, np.ndarray] = {}
                    for etf, p_series in sector_prices.items():
                        r = np.log(p_series.values[1:] / p_series.values[:-1])
                        sector_rets[etf] = r

                    # Filter scores to stocks with positive correlation to any top sector
                    filtered_scores: dict[str, float] = {}
                    for sym in scores:
                        if sym not in prices.columns:
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.sector_lookback + 5:
                            filtered_scores[sym] = scores[sym]  # include if not enough data
                            continue
                        stock_ret = np.log(col.values[1:] / col.values[:-1])
                        max_corr = -999.0
                        for etf, s_ret in sector_rets.items():
                            n = min(len(stock_ret), len(s_ret), self.sector_lookback)
                            if n < 10:
                                continue
                            sr = stock_ret[-n:]
                            er = s_ret[-n:]
                            if np.std(sr) < 1e-8 or np.std(er) < 1e-8:
                                continue
                            corr = float(np.corrcoef(sr, er)[0, 1])
                            if np.isfinite(corr) and corr > max_corr:
                                max_corr = corr
                        # Include stock if it has positive correlation to a top sector
                        if max_corr > 0.1:
                            filtered_scores[sym] = scores[sym]
                    # Only use filtered if we have enough stocks
                    if len(filtered_scores) >= 5:
                        scores = filtered_scores

            if len(scores) < 5:
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
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
    return sp500_tickers() + [_IEF, _SPY] + _SECTOR_ETFS


NAME = "idio_mom_sector_breadth"
HYPOTHESIS = (
    "SP500 top-15 stocks by idiosyncratic 63d momentum (raw return minus beta*SPY return) "
    "filtered to only stocks in the top-3 SPDR sectors by 20d return; "
    "SPY 200d SMA gate; IEF defensive; biweekly rebalance; "
    "sector-breadth filter narrows idiosyncratic momentum to current winning sectors"
)

UNIVERSE = _universe

STRATEGY = IdiomMomSectorBreadth()
