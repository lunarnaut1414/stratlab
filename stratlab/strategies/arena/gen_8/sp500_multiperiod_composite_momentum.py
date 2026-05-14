"""SP500 Multi-Timeframe Composite Momentum — gen_8 sonnet-9

Hypothesis: Rank SP500 stocks by the AVERAGE PERCENTILE RANK across three
momentum windows (21d, 63d, 126d). Hold top-15 stocks that also satisfy:
- Each stock is above its own 200d SMA (individual trend filter)
- SPY is above its 200d SMA (market trend gate)

When SPY is below 200d SMA, hold IEF (intermediate bonds) as defensive.
Equal-weight positions. Biweekly rebalance (every 10 bars).

Rationale: Single-window momentum (e.g., 63d) can capture stocks that are
temporarily hot due to sector rotation or earnings surprises. A composite
rank across 3 time horizons selects stocks with PERSISTENT multi-horizon
momentum — stocks that rank well at 1-month, 3-month, AND 6-month horizons
simultaneously. This selects genuinely trending names rather than noise-driven
winners, and should produce more stable selection turnover.

Differentiation: The leaderboard has 63d, 42d, 126d, and 21d single-window
strategies, but NO composite rank aggregation across multiple windows. The
use of per-stock 200d SMA filter (like gen7_126d_goldencross) adds a local
trend confirmation. IEF defensive (not TLT) reduces duration sensitivity.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

WINDOWS = [21, 63, 126]    # Momentum lookback windows
STOCK_TREND = 200          # Individual stock SMA filter
TREND_WINDOW = 200         # SPY 200d SMA market gate
TOP_K = 15
REBALANCE_DAYS = 10        # Biweekly
EXPOSURE = 0.97
MAX_LOOKBACK = max(WINDOWS)


def _universe() -> list[str]:
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe


class Sp500MultiperiodCompositeMomentum(Strategy):
    """Top-15 SP500 stocks by composite 21d+63d+126d percentile-rank momentum."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, MAX_LOOKBACK) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live = {s: float(closes[s]) for s in closes.index
                if closes[s] > 0 and s not in ("SPY", "IEF")}

        # --- SPY 200d SMA trend gate ---
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < TREND_WINDOW:
            return []
        spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
        spy_price = float(closes.get("SPY", 0.0)) if "SPY" in closes.index else 0.0
        spy_bull = spy_price > spy_sma

        if not spy_bull:
            # Bear: hold IEF
            target = {"IEF": EXPOSURE}
        else:
            # Compute multi-window momentum for all SP500 stocks
            prices_window = ctx.closes_window(MAX_LOOKBACK + 10)
            if len(prices_window) < MAX_LOOKBACK:
                return []

            # Compute raw returns for each window
            returns_by_window: dict[int, dict[str, float]] = {w: {} for w in WINDOWS}
            for sym in live:
                if sym not in prices_window.columns:
                    continue
                col = prices_window[sym].dropna()
                for w in WINDOWS:
                    if len(col) < w:
                        continue
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-w])
                    if p_start > 0:
                        r = p_end / p_start - 1.0
                        if np.isfinite(r):
                            returns_by_window[w][sym] = r

            # Compute percentile ranks for each window
            candidate_symbols = set(live.keys())
            for w in WINDOWS:
                candidate_symbols &= set(returns_by_window[w].keys())

            if len(candidate_symbols) < TOP_K:
                target = {"IEF": EXPOSURE}
            else:
                # Rank within each window
                composite: dict[str, float] = {}
                for w in WINDOWS:
                    rets = {s: returns_by_window[w][s] for s in candidate_symbols}
                    sorted_syms = sorted(rets.keys(), key=lambda x: rets[x])
                    n = len(sorted_syms)
                    for rank, sym in enumerate(sorted_syms):
                        if sym not in composite:
                            composite[sym] = 0.0
                        # Add percentile rank (0 to 1)
                        composite[sym] += rank / max(n - 1, 1)

                # Average percentile rank across windows
                avg_composite = {sym: v / len(WINDOWS) for sym, v in composite.items()}

                # Sort by composite score descending
                ranked = sorted(avg_composite.items(), key=lambda x: x[1], reverse=True)

                # Apply per-stock 200d SMA filter on top candidates
                selected = []
                for sym, _ in ranked:
                    if len(selected) >= TOP_K:
                        break
                    hist = ctx.history(sym)
                    if len(hist) < STOCK_TREND:
                        continue
                    sma = float(hist["close"].iloc[-STOCK_TREND:].mean())
                    price = live.get(sym, 0.0)
                    if price > sma:
                        selected.append(sym)

                if not selected:
                    target = {"IEF": EXPOSURE}
                else:
                    target = {sym: EXPOSURE / len(selected) for sym in selected}

        # Compute portfolio equity
        all_prices = {s: float(closes[s]) for s in closes.index if closes[s] > 0}
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = all_prices.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = all_prices.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "sp500_multiperiod_composite_momentum"
HYPOTHESIS = (
    "SP500 multi-timeframe composite momentum: rank stocks by average percentile rank across "
    "21d+63d+126d returns, hold top-15 above 200d SMA; SPY 200d SMA gate to IEF defensive; "
    "equal-weight biweekly rebalance; composite rank selects stocks with persistent multi-horizon momentum."
)

STRATEGY = Sp500MultiperiodCompositeMomentum()
