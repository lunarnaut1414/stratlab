"""Sector Sharpe-ratio rotation strategy.

Hypothesis: Rank 9 SPDR sector ETFs by their rolling 63-day Sharpe ratio
(annualized daily return / annualized daily vol). Hold top-2 equally when
SPY is in a bull market (above 200d SMA). Rotate to TLT when bearish.
Rebalance every 5 bars (weekly).

Rationale: Sharpe-ranked rotation selects sectors that have the best
RISK-ADJUSTED momentum, not just absolute return momentum. This tends to
favor sectors with smoother trends (lower vol) over high-momentum but
volatile sectors. The combination of return AND volatility creates a
different ranking than pure price momentum or low-vol factor alone.

Key structural differences:
- Sharpe ratio (return/vol) not absolute return or inverse-vol weighting
- 9-sector universe, top-2 holdings (vs 3 in sector_relative_momentum)
- 63d lookback (shorter than typical 6-month momentum)
- Different from lowvol_factor (selects stocks by low vol, not sectors by Sharpe)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY"]
UNIVERSE = SECTORS + ["SPY", "TLT"]

SHARPE_WINDOW = 63    # 63-day Sharpe window
TREND_WINDOW = 200    # SPY 200d SMA
REBALANCE_EVERY = 5   # weekly
TOP_K = 2             # hold top-2 sectors
EXPOSURE = 0.97


def _rolling_sharpe(returns: pd.Series, window: int) -> float:
    """Compute annualized Sharpe ratio over `window` days."""
    if len(returns) < window:
        return 0.0
    r = returns.iloc[-window:]
    mean = float(r.mean())
    std = float(r.std(ddof=1))
    if std < 1e-9:
        return 0.0
    # Annualize: 252 trading days
    return float(mean / std * np.sqrt(252))


class SectorSharpeRotation(Strategy):
    def __init__(
        self,
        sharpe_window: int = SHARPE_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            sharpe_window=sharpe_window,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.sharpe_window = int(sharpe_window)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.sharpe_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY trend filter
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Compute Sharpe ratios for each sector
            sharpe_scores: dict[str, float] = {}
            for sym in SECTORS:
                try:
                    hist = ctx.history(sym)
                except KeyError:
                    continue
                if hist is None or len(hist) < self.sharpe_window + 2:
                    continue
                close = hist["close"].dropna()
                if len(close) < self.sharpe_window + 1:
                    continue
                # Log returns
                returns = close.pct_change().dropna()
                if len(returns) < self.sharpe_window:
                    continue
                sharpe = _rolling_sharpe(returns, self.sharpe_window)
                if np.isfinite(sharpe):
                    sharpe_scores[sym] = sharpe

            if len(sharpe_scores) < self.top_k:
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                k = min(self.top_k, len(sharpe_scores))
                ranked = sorted(sharpe_scores, key=sharpe_scores.__getitem__, reverse=True)[:k]
                w = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
                        target[sym] = w

                if not target and "SPY" in live:
                    target["SPY"] = self.exposure

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sector_sharpe_rotation"
HYPOTHESIS = (
    "SP500 sector Sharpe momentum: rank 9 SPDR sector ETFs by rolling 63d Sharpe "
    "ratio (daily return / daily vol); hold top-2 sectors equally when SPY above "
    "200d SMA; TLT when bearish; weekly rebalance; Sharpe-ranked sector rotation "
    "captures both return AND risk-adjusted momentum"
)

STRATEGY = SectorSharpeRotation()
