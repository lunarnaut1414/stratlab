"""Long-End Yield Slope Gated QQQ/IEF Router — gen_9 sonnet-7

Hypothesis: The 30Y-10Y long-end yield curve slope (TYX minus TNX) vs its 200d
MA as a duration-risk-premium regime signal routing to QQQ vs IEF/TLT.

Regime logic:
  - SPY below 200d SMA → TLT 97% (outer equity bear gate)
  - SPY above 200d SMA AND TYX-TNX above its 200d MA (steep long-end = growth
    regime, duration premium supporting risk assets) → QQQ 97%
  - SPY above 200d SMA AND TYX-TNX below its 200d MA (flat/inverted long-end =
    recession/stagflation risk) → IEF 60%+TLT 37% (defensive bond blend)

Rationale: gen8_opus1_longend_slope_equity_gate is the #1 OOS performer across
all rounds (OOS Calmar 0.79, retaining 95% of IS Calmar 0.83). That strategy
routes to SP500 momentum stocks in the favorable regime. This variant routes to
QQQ instead — capturing the same macro signal's favorable-regime alpha via
tech/growth ETF rather than individual SP500 stocks. The pure ETF approach has
lower expected corr to the existing SP500-momentum cluster.

Distinct from gen9_sonnet1_idio_momentum_longend_slope: that uses SP500 stock
selection with idiosyncratic momentum; this uses pure QQQ/IEF ETF routing.
The corr profile should differ substantially (QQQ vs SP500 individual stocks).

Weekly rebalance for adequate trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ── Parameters ──────────────────────────────────────────────────────────────
SLOPE_MA_WINDOW = 200    # MA of TYX-TNX spread
SPY_TREND = 200          # SPY outer bear gate
REBALANCE_DAYS = 5       # weekly rebalance
EXPOSURE = 0.97

# Regime allocations
RISK_ON = [("QQQ", 0.97)]
RISK_OFF = [("IEF", 0.60), ("TLT", 0.37)]
BEAR = [("TLT", 0.97)]


class LongEndSlopeQqqRouter(Strategy):
    """TYX-TNX slope vs 200d MA: QQQ risk-on, IEF+TLT risk-off, TLT bear."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = SLOPE_MA_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []
        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # SPY outer bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma200 = float(spy_close.iloc[-SPY_TREND:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma200

        if not spy_bull:
            regime_holdings = BEAR
        else:
            # Long-end yield curve slope signal
            try:
                tyx_hist = ctx.history("^TYX")
                tnx_hist = ctx.history("^TNX")
            except KeyError:
                # If signal unavailable, neutral allocation
                regime_holdings = RISK_ON
            else:
                if len(tyx_hist) < SLOPE_MA_WINDOW + 5 or len(tnx_hist) < SLOPE_MA_WINDOW + 5:
                    return []

                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()

                # Align on common length
                min_len = min(len(tyx_close), len(tnx_close))
                if min_len < SLOPE_MA_WINDOW + 5:
                    return []

                tyx_arr = tyx_close.iloc[-min_len:].values
                tnx_arr = tnx_close.iloc[-min_len:].values
                slope = tyx_arr - tnx_arr  # 30Y - 10Y spread

                current_slope = float(slope[-1])
                slope_ma = float(np.mean(slope[-SLOPE_MA_WINDOW:]))

                if current_slope > slope_ma:
                    # Steep long-end: growth-supportive, go risk-on
                    regime_holdings = RISK_ON
                else:
                    # Flat/inverted long-end: duration stress, go defensive bonds
                    regime_holdings = RISK_OFF

        targets = dict(regime_holdings)

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in targets and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
        for sym, weight in targets.items():
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
    return ["SPY", "QQQ", "IEF", "TLT", "^TYX", "^TNX"]


NAME = "gen9_longend_slope_qqq_router"
HYPOTHESIS = (
    "TYX-TNX long-end slope vs 200d MA as duration-premium gate for QQQ: "
    "slope > 200d MA (steep long-end, growth-supportive) -> QQQ 97%; "
    "slope < 200d MA (flat/inverted) -> IEF 60%+TLT 37%; "
    "SPY bear -> TLT 97%. Weekly rebalance. ETF-only, no individual stocks."
)

UNIVERSE = _universe

STRATEGY = LongEndSlopeQqqRouter()
