"""Sector-Neutral SP500 Momentum — gen_8 sonnet-8

Hypothesis: Rank SP500 stocks by 42d return *within* each GICS sector;
take top-3 from each sector (up to 9 sectors active); equal-weight.
SPY 200d SMA gate → IEF defensive. Biweekly rebalance.

Rationale: Pure cross-sectional momentum concentrates in the hottest
sector of the IS window (e.g. financials post-2010, tech 2013-2018).
Sector-neutral selection forces diversification across 9-11 GICS sectors,
producing a portfolio structurally distinct from existing SP500 momentum
strategies on the leaderboard. The within-sector ranking captures the
idiosyncratic winner in each segment — not just the entire hot sector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
REBALANCE_EVERY = 10          # ~biweekly
MOMENTUM_WINDOW = 42          # ~2 months
TREND_WINDOW = 200            # SPY bear gate
TOP_K_PER_SECTOR = 3          # picks per sector
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"

# GICS sector slugs we target (excludes "other" / uncategorized)
_TARGET_SECTORS = {
    "information_technology",
    "health_care",
    "financials",
    "consumer_discretionary",
    "industrials",
    "communication_services",
    "consumer_staples",
    "energy",
    "materials",
    "real_estate",
    "utilities",
}


def _load_sector_map() -> dict[str, str]:
    """Load ticker→sector from the on-disk catalog.json (no network call)."""
    candidate = Path(__file__).resolve().parents[5] / "data" / "market" / "catalog.json"
    if not candidate.exists():
        return {}
    try:
        catalog = json.loads(candidate.read_text())
        stocks = catalog.get("stocks", {})
        return {t: v["sector"] for t, v in stocks.items() if "sector" in v}
    except Exception:
        return {}


# Load once at module import (cached for the entire run)
_SECTOR_MAP: dict[str, str] = _load_sector_map()


class SectorNeutralSP500Momentum(Strategy):
    """Equal-weight top-3 per GICS sector by 42d momentum; SPY trend gate."""

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
        warmup = self.trend_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # ---- SPY trend gate ----
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
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
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Compute 42d momentum
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            # Score each symbol
            scores: dict[str, float] = {}
            for sym in prices.columns:
                sector = _SECTOR_MAP.get(sym)
                if sector not in _TARGET_SECTORS:
                    continue
                if sym not in live:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                if _IEF in live:
                    target[_IEF] = self.exposure
                # else hold cash implicitly
            else:
                # Group by sector, take top-K per sector
                sector_scores: dict[str, list[tuple[str, float]]] = {}
                for sym, sc in scores.items():
                    sect = _SECTOR_MAP[sym]
                    sector_scores.setdefault(sect, []).append((sym, sc))

                selected: list[str] = []
                for sect, pairs in sector_scores.items():
                    pairs.sort(key=lambda x: x[1], reverse=True)
                    top = [s for s, _ in pairs[:self.top_k_per_sector]]
                    selected.extend(top)

                if not selected:
                    if _IEF in live:
                        target[_IEF] = self.exposure
                else:
                    per_weight = self.exposure / len(selected)
                    for sym in selected:
                        if sym in live:
                            target[sym] = per_weight

        # ---- Build orders ----
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
    return sp500_tickers() + [_IEF, _SPY]


NAME = "sector_neutral_sp500_momentum"
HYPOTHESIS = (
    "Sector-neutral SP500 momentum: within each GICS sector, rank stocks by 42d return; "
    "take top-3 per sector (up to 9 sectors active); equal-weight within sector; "
    "SPY 200d SMA gate; IEF defensive; biweekly rebalance; avoids sector concentration "
    "that pure cross-sectional momentum creates"
)

UNIVERSE = _universe

STRATEGY = SectorNeutralSP500Momentum()
