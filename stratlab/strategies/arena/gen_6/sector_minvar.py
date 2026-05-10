"""Sector minimum-variance portfolio strategy.

Hypothesis: When SPY is in a bull market (above 200d SMA), hold a minimum-
variance weighted portfolio of 9 SPDR sector ETFs. When SPY is below the
200d SMA, rotate to TLT (defensive). Monthly rebalance.

Rationale: Minimum-variance (min-var) portfolios consistently achieve lower
realized volatility than equal-weight or cap-weight, while capturing similar
returns over long horizons. By applying min-var to sector ETFs (rather than
individual stocks), we get sector diversification with reduced drawdowns.

This is structurally different from all leaderboard strategies because:
- No momentum signal - min-var purely minimizes variance
- 9 sector ETFs provide even diversification across the economy
- Covariance-based weighting vs inverse-vol or equal-weight
- Sectors rebalance only monthly, reducing correlation to high-turnover strategies

Min-var optimization: minimize w'Σw subject to sum(w)=1, w>=min_weight.
Solved analytically via Σ^{-1} * 1 / (1' * Σ^{-1} * 1) with clipping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SECTORS = ["XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY"]
UNIVERSE = SECTORS + ["SPY", "TLT"]

COV_WINDOW = 60       # 60-day covariance
TREND_WINDOW = 200    # SPY 200d SMA
REBALANCE_EVERY = 21  # monthly
EXPOSURE = 0.97
MIN_WEIGHT = 0.03     # minimum weight per sector (prevent extreme concentration)
MAX_WEIGHT = 0.35     # maximum weight per sector


class SectorMinVar(Strategy):
    def __init__(
        self,
        cov_window: int = COV_WINDOW,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
    ) -> None:
        super().__init__(
            cov_window=cov_window,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        self.cov_window = int(cov_window)
        self.trend_window = int(trend_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def _min_var_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        """Compute minimum variance weights via inverse-covariance.

        Analytical solution: w* = Σ^{-1}*1 / (1'*Σ^{-1}*1)
        Then clip to [min_weight, max_weight] and renormalize.
        """
        cov = returns.cov().values
        n = cov.shape[0]
        syms = list(returns.columns)

        # Add regularization to ensure invertibility
        reg = 1e-6
        cov_reg = cov + np.eye(n) * reg

        try:
            inv_cov = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            return {s: 1.0 / n for s in syms}

        ones = np.ones(n)
        raw = inv_cov @ ones
        denom = ones @ raw
        if abs(denom) < 1e-10:
            return {s: 1.0 / n for s in syms}

        raw_weights = raw / denom

        # Clip to bounds
        clipped = np.clip(raw_weights, self.min_weight, self.max_weight)
        # Renormalize
        total = clipped.sum()
        if total <= 0:
            return {s: 1.0 / n for s in syms}
        normalized = clipped / total

        return {syms[i]: float(normalized[i]) for i in range(n)}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.cov_window, self.trend_window) + 5
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
            # Bear market: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Bull market: sector min-var
            prices_window = ctx.closes_window(self.cov_window + 2)
            if len(prices_window) < self.cov_window:
                return []

            # Filter to available sectors
            available_sectors = [
                s for s in SECTORS
                if s in prices_window.columns and prices_window[s].dropna().shape[0] >= self.cov_window
            ]

            if len(available_sectors) < 3:
                # Fallback: equal-weight available sectors
                if available_sectors:
                    w = self.exposure / len(available_sectors)
                    target = {s: w for s in available_sectors}
                elif "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                # Compute returns
                sector_prices = prices_window[available_sectors].iloc[-self.cov_window - 1:]
                # Use log returns for covariance
                returns = np.log(sector_prices / sector_prices.shift(1)).dropna()
                if len(returns) < self.cov_window // 2:
                    # Not enough data: fallback to equal weight
                    w = self.exposure / len(available_sectors)
                    target = {s: w for s in available_sectors}
                else:
                    weights = self._min_var_weights(returns)
                    for sym, w in weights.items():
                        target[sym] = w * self.exposure

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


NAME = "sector_minvar"
HYPOTHESIS = (
    "Sector minimum-variance portfolio: when SPY above 200d SMA hold minimum-variance "
    "weighted portfolio of 9 SPDR sector ETFs (XLK/XLF/XLV/XLI/XLP/XLU/XLE/XLB/XLY) "
    "using 60d covariance matrix; when SPY below 200d SMA hold TLT; monthly rebalance; "
    "min-variance weighting orthogonal to momentum and VIX signals"
)

STRATEGY = SectorMinVar()
