"""SPDR Sector ETF Rotation by Sortino Ratio — gen_8 sonnet-3

Hypothesis: Rank all 11 SPDR sector ETFs by rolling 63d Sortino ratio
(return / downside-vol), hold top-3 equal-weight when SPY above 150d SMA;
TLT defensive when SPY below; weekly rebalance.

Rationale: Pure momentum sector rotation (ranked by return) already exists
in the leaderboard. Sortino scoring selects sectors with both high returns AND
low downside volatility — preferring sectors with clean uptrends rather than
high-volatility leaders. This is conceptually distinct from Sharpe (which uses
total vol), momentum (which uses raw return), and low-vol (which ignores return).
The 150d SMA gate (vs the more common 200d) provides slightly earlier entry/exit.

IS window: 2010-2018 (9 years).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5            # Weekly
SORTINO_WINDOW = 63            # ~3 months for Sortino scoring
TREND_WINDOW = 150             # SPY 150d SMA gate
TOP_K = 3
EXPOSURE = 0.97

SECTOR_ETFS = [
    "XLK",   # Technology
    "XLV",   # Health Care
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLE",   # Energy
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLY",   # Consumer Discretionary
    "XLC",   # Communication Services
]


def _compute_sortino(prices: np.ndarray, window: int) -> float:
    """Compute Sortino ratio (annualized) over the last `window` bars."""
    if len(prices) < window + 1:
        return -999.0
    tail = prices[-window - 1:]
    log_rets = np.log(tail[1:] / tail[:-1])
    mean_ret = float(np.mean(log_rets))
    # Downside deviation: std of negative returns only
    neg_rets = log_rets[log_rets < 0]
    if len(neg_rets) == 0:
        # No negative returns — extremely strong uptrend; use small epsilon
        downside_vol = 1e-6
    else:
        downside_vol = float(np.std(neg_rets))
    if downside_vol < 1e-8:
        return 999.0  # effectively infinite Sortino
    # Annualize (252 trading days)
    ann_ret = mean_ret * 252
    ann_downside = downside_vol * np.sqrt(252)
    return ann_ret / ann_downside


class SortinoSectorRotation(Strategy):
    """Rank SPDR sector ETFs by 63d Sortino ratio; hold top-3; SPY 150d gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        sortino_window: int = SORTINO_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            sortino_window=sortino_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.sortino_window = int(sortino_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.sortino_window + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

        # Check SPY 150d SMA for regime gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Defensive: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Rank sector ETFs by Sortino ratio
            need = self.sortino_window + 5
            prices_df = ctx.closes_window(need)

            scores: dict[str, float] = {}
            for sym in SECTOR_ETFS:
                if sym not in prices_df.columns:
                    continue
                col = prices_df[sym].dropna()
                if len(col) < self.sortino_window + 1:
                    continue
                sortino = _compute_sortino(col.values, self.sortino_window)
                if np.isfinite(sortino):
                    scores[sym] = sortino

            if not scores:
                return []

            # Hold top-k by Sortino (must have positive Sortino = positive returns)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected = [sym for sym, s in ranked[:self.top_k] if s > 0]

            if not selected:
                # Nothing has positive Sortino; go defensive
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                per_slot = self.exposure / len(selected)
                for sym in selected:
                    target[sym] = per_slot

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


NAME = "sortino_sector_rotation"
HYPOTHESIS = (
    "SP500 Sortino-ratio-scored sector ETF rotation: rank all 11 SPDR sector ETFs "
    "(XLK,XLV,XLF,XLI,XLP,XLU,XLE,XLB,XLRE,XLY,XLC) by rolling 63d Sortino ratio "
    "(return/downside-vol), hold top-3 equal-weight when SPY above 150d SMA; "
    "TLT defensive when SPY below; weekly rebalance; downside-risk-adjusted scoring "
    "selects sectors with high returns AND low drawdowns vs pure-momentum"
)

UNIVERSE = SECTOR_ETFS + ["SPY", "TLT", "SHY"]

STRATEGY = SortinoSectorRotation()
