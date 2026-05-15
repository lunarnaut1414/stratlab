"""Sector breadth gate on SP500 momentum.

Hypothesis: Use the count of SPDR sector ETFs above their 50d SMA as a
market breadth signal — when most sectors participate in an uptrend, the
regime is healthy for equity momentum; when fewer than half are in uptrend,
the regime is deteriorating or sideways.

Regime tiers (out of 11 XL* sectors):
  - Breadth >= 7/11 (broad participation): hold top-15 SP500 stocks by 63d
    momentum, equally weighted at 97% exposure.
  - Breadth 4-6/11 (mixed market): hold IEF (intermediate treasuries) at 97%.
  - Breadth < 4/11 (broad sector weakness): hold TLT (long bonds) at 97%.
  - Override: if SPY below 200d SMA, always hold TLT regardless of breadth.

Rationale: Breadth measures aggregate sector participation rather than a
single macro indicator (VIX, yield, credit). A market can be rising on
narrow leadership even while most sectors weaken — breadth detects this.
The 3-tier design is structurally different from binary VIX/credit gates.

Distinct from all existing strategies: no existing leaderboard entry uses
sector-ETF SMA-above-count as a regime signal.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# 9 SPDR sector ETFs available from 2010 (XLRE launched 2015, XLC launched 2018 — exclude)
SECTOR_ETFS = [
    "XLK", "XLV", "XLF", "XLE", "XLI",
    "XLU", "XLY", "XLP", "XLB",
]

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 63     # ~3 months
BREADTH_SMA = 50         # 50d SMA for breadth scoring
SPY_TREND_WINDOW = 200   # outer bear gate
TOP_K = 15
HIGH_BREADTH = 6         # >= 6/9 sectors -> risk-on
LOW_BREADTH = 3          # < 3/9 -> TLT; 3-5 -> IEF
EXPOSURE = 0.97


class SectorBreadthGateSP500(Strategy):
    """Sector breadth regime gate driving SP500 momentum allocation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        breadth_sma: int = BREADTH_SMA,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        high_breadth: int = HIGH_BREADTH,
        low_breadth: int = LOW_BREADTH,
        exposure: float = EXPOSURE,
        n_sectors: int = len(SECTOR_ETFS),
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            breadth_sma=breadth_sma,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            high_breadth=high_breadth,
            low_breadth=low_breadth,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.breadth_sma = int(breadth_sma)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.high_breadth = int(high_breadth)
        self.low_breadth = int(low_breadth)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.breadth_sma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma200 = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        # --- Sector breadth score ---
        breadth_count = 0
        sectors_scored = 0
        for sector in SECTOR_ETFS:
            try:
                sec_hist = ctx.history(sector)
            except KeyError:
                continue
            if sec_hist is None or len(sec_hist) < self.breadth_sma + 2:
                continue
            sec_close = sec_hist["close"].dropna()
            if len(sec_close) < self.breadth_sma:
                continue
            sma = float(sec_close.iloc[-self.breadth_sma:].mean())
            current = float(sec_close.iloc[-1])
            if np.isfinite(sma) and np.isfinite(current) and sma > 0:
                sectors_scored += 1
                if current > sma:
                    breadth_count += 1

        # Need at least 5 sectors to make a reliable breadth call
        if sectors_scored < 5:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Determine regime
        if not spy_bull:
            # Bear market override -> full TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif breadth_count >= self.high_breadth:
            # Risk-on: top-K SP500 momentum
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []
            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret
            if len(scores) < self.top_k:
                # Not enough candidates — fall back to IEF
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                longs = ranked[:self.top_k]
                per_weight = self.exposure / len(longs)
                for sym in longs:
                    target[sym] = per_weight
        elif breadth_count < self.low_breadth:
            # Broad sector weakness -> TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Mixed (4-6/11 sectors above 50d SMA) -> IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Sell positions not in target
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
    return (
        sp500_tickers()
        + ["TLT", "IEF", "SPY"]
        + SECTOR_ETFS
    )


NAME = "sector_breadth_gate_sp500"
HYPOTHESIS = (
    "Sector breadth gate on SP500 momentum: count of XL* sector ETFs above 50d SMA "
    "as breadth score; hold top-15 SP500 stocks by 63d momentum when breadth score >= 7/11 "
    "sectors; hold IEF when score 4-6 (mixed); hold TLT when score < 4 (broad sector weakness); "
    "SPY 200d outer gate; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SectorBreadthGateSP500()
