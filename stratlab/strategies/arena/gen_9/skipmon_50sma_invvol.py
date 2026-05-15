"""SP500 Skip-Month Momentum + Per-Stock 50d SMA + Inverse-Vol Weighting — gen_9 sonnet-2

Hypothesis: The Jegadeesh-Titman (1993) skip-month momentum factor (12-1 months,
i.e., look back 126 days but skip the most recent 21 days) avoids short-term
reversal contamination in the momentum signal. Combining this with:
  1. Per-stock 50d SMA trend gate (shorter than gen_8's 63d SMA variant)
     to filter for recently-trending rather than merely long-term-uptrending names.
  2. Inverse realized-vol position sizing to reduce concentration in high-beta winners.
  3. SPY 150d SMA market gate (intermediate trend, distinct from the 200d used by most
     existing strategies).

Differentiation from leaderboard:
  - gen8_sp500_skipmon_63sma_momentum (IS 0.80): uses 63d stock SMA + equal weight +
    200d SPY market gate. This strategy uses 50d stock SMA (more recent trend signal)
    + inverse-vol weighting + 150d SPY market gate.
  - gen7_sp500_126d_stock_50sma_goldencross: uses 126d momentum but with a golden-cross
    SPY gate (50d vs 150d) and equal weighting. This strategy uses skip-month (126d-21d)
    and adds inverse-vol sizing.
  - The combination of skip-month + 50d stock SMA + inverse-vol + 150d SPY gate has
    not been tried in any prior round.

Why inverse-vol weighting: momentum strategies can be dominated by high-momentum,
high-volatility names that later crash. Inverse-vol weighting gives more capital
to the steady outperformers and less to volatile ones, historically improving
the risk-adjusted profile without sacrificing much raw return.

IEF as the defensive asset (intermediate duration, less rate sensitivity than TLT).

IS window: 2010-2018, biweekly rebalance (every 10 bars) for trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOM_LONG = 126            # Skip-month lookback: 126 days
MOM_SKIP = 21             # Skip most recent 21 days
STOCK_SMA = 50            # Per-stock 50d SMA trend filter
TREND_WINDOW = 150        # SPY 150d SMA market gate (distinct from 200d)
VOL_WINDOW = 21           # Realized vol window for weighting
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe


class SkipmonSmaInvvol(Strategy):
    """SP500 skip-month momentum with 50d SMA filter, inverse-vol weights, 150d SPY gate."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, MOM_LONG) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 150d SMA market gate ---
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target = {"IEF": EXPOSURE}
        else:
            # --- Skip-month momentum on SP500 stocks ---
            prices_window = ctx.closes_window(MOM_LONG + 5)
            if len(prices_window) < MOM_LONG:
                return []

            live = {s: float(closes[s]) for s in closes.index
                    if closes[s] > 0 and s not in ("IEF", "SPY")}

            scores: dict[str, float] = {}
            vols: dict[str, float] = {}

            for sym in live:
                if sym not in prices_window.columns:
                    continue
                col = prices_window[sym].dropna()
                if len(col) < MOM_LONG:
                    continue

                # Skip-month: 126d return excluding last 21 days
                p_end_skip = float(col.iloc[-MOM_SKIP - 1])   # price 21 days ago
                p_start = float(col.iloc[-MOM_LONG])          # price 126 days ago
                if p_start <= 0 or p_end_skip <= 0:
                    continue
                r = p_end_skip / p_start - 1.0
                if not np.isfinite(r):
                    continue

                # Per-stock 50d SMA trend filter
                if len(col) < STOCK_SMA:
                    continue
                sma_50 = float(col.iloc[-STOCK_SMA:].mean())
                current_price = live.get(sym, 0.0)
                if current_price <= 0 or current_price <= sma_50:
                    continue  # skip stocks below 50d SMA

                # Realized volatility for inverse-vol weighting
                if len(col) < VOL_WINDOW + 1:
                    continue
                rets = np.diff(np.log(col.values[-VOL_WINDOW - 1:]))
                rv = float(np.std(rets))
                if rv <= 0 or not np.isfinite(rv):
                    continue

                scores[sym] = r
                vols[sym] = rv

            if len(scores) < TOP_K:
                # Fall back to IEF if not enough qualifying stocks
                target = {"IEF": EXPOSURE}
            else:
                # Rank by skip-month momentum and take top-K
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                selected = ranked[:TOP_K]

                # Inverse-vol weighting
                inv_vols = {sym: 1.0 / vols[sym] for sym in selected if sym in vols}
                if not inv_vols:
                    target = {"IEF": EXPOSURE}
                else:
                    total_inv_vol = sum(inv_vols.values())
                    target = {sym: (inv_vols[sym] / total_inv_vol) * EXPOSURE
                              for sym in inv_vols}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
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


NAME = "skipmon_50sma_invvol"
HYPOTHESIS = (
    "SP500 skip-month momentum (126d-skip-21d) with per-stock 50d SMA trend gate "
    "and inverse-vol position weighting; SPY 150d SMA market gate; IEF defensive; "
    "hold top-15 stocks qualifying on skip-month score AND price above 50d SMA; "
    "biweekly rebalance; distinct from gen8 skip-month by 50d SMA + inv-vol + 150d gate."
)

STRATEGY = SkipmonSmaInvvol()
