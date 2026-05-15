"""Long-end yield-curve slope gating SP500 momentum vs sector ETF defensive.

Hypothesis (sonnet-5, gen_9):
    TYX-TNX slope direction vs 200d MA gates SP500 individual stock momentum:
    - Slope rising above its 200d MA (steepening = growth expectations rising)
      -> top-15 SP500 stocks by 63d momentum, SPY 200d SMA outer gate
    - Slope flat/falling below 200d MA (flattening = recession risk)
      -> top-3 defensive sector ETFs (XLU, XLV, XLP) by 42d return
    - SPY bear (SPY < 200d SMA) -> TLT 60% + IEF 37%
    Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen8_opus1_longend_slope_equity_gate: uses TYX-TNX vs its own 200d MA to
    gate TOP-15 SP500 momentum vs SPY+TLT blend. THIS strategy:
    - Uses the slope's momentum direction (above/below its 200d MA) same way
    - But routes defensively to TOP-3 DEFENSIVE SECTORS (XLU/XLV/XLP) not SPY+TLT
    - The defensive routing to sector ETFs changes the PnL path materially
    - 3 tiers (aggressive stock / defensive sector / full bond) vs 2 tiers
  - Different from any existing sector rotation strategies which use VIX/JNK/TNX gates.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
TREND_WINDOW = 200          # SPY 200d SMA gate; also slope 200d MA
MOM_WINDOW_STOCK = 63       # SP500 stock momentum window
MOM_WINDOW_SECTOR = 42      # defensive sector momentum window
TOP_K_STOCK = 15            # SP500 stocks in growth regime
TOP_K_SECTOR = 3            # defensive sectors in flat-curve regime
EXPOSURE = 0.97

# Defensive sectors for slope-flat regime
DEFENSIVE_SECTORS = ["XLU", "XLV", "XLP", "XLI"]


class LongendSlopeSectorRotation(Strategy):
    """TYX-TNX slope 200d MA gating stock momentum vs defensive sectors vs bonds."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = TREND_WINDOW + MOM_WINDOW_STOCK + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # --- Compute TYX-TNX slope and its 200d MA ---
        try:
            tyx_hist = ctx.history("^TYX")
            tnx_hist = ctx.history("^TNX")
        except KeyError:
            return []

        tyx_close = tyx_hist["close"].dropna()
        tnx_close = tnx_hist["close"].dropna()

        min_len = min(len(tyx_close), len(tnx_close))
        if min_len < TREND_WINDOW + 5:
            return []

        # Align and compute slope series
        tyx_arr = tyx_close.values[-min_len:]
        tnx_arr = tnx_close.values[-min_len:]
        slope_series = tyx_arr - tnx_arr
        current_slope = float(slope_series[-1])
        slope_200d_ma = float(slope_series[-TREND_WINDOW:].mean())
        slope_steepening = current_slope > slope_200d_ma

        # --- SPY bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-TREND_WINDOW:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # SPY bear -> TLT + IEF defensive
            if "TLT" in closes_now.index:
                target["TLT"] = 0.60
            if "IEF" in closes_now.index:
                target["IEF"] = 0.37
        elif slope_steepening:
            # Slope steepening -> SP500 cross-sectional momentum
            need = MOM_WINDOW_STOCK + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOM_WINDOW_STOCK:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_WINDOW_STOCK:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_WINDOW_STOCK])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                scores[sym] = p_end / p_start - 1.0

            if len(scores) < 5:
                # Not enough candidates
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K_STOCK]
                per_slot = EXPOSURE / len(ranked)
                for sym in ranked:
                    target[sym] = per_slot
        else:
            # Slope flattening -> defensive sector ETFs
            prices = ctx.closes_window(MOM_WINDOW_SECTOR + 5)
            if len(prices) < MOM_WINDOW_SECTOR:
                return []

            sector_scores: dict[str, float] = {}
            for sym in DEFENSIVE_SECTORS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < MOM_WINDOW_SECTOR:
                    continue
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_WINDOW_SECTOR])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                sector_scores[sym] = p_end / p_start - 1.0

            if not sector_scores:
                if "TLT" in closes_now.index:
                    target["TLT"] = EXPOSURE
            else:
                ranked = sorted(sector_scores, key=sector_scores.__getitem__, reverse=True)
                selected = ranked[:TOP_K_SECTOR]
                per_slot = EXPOSURE / len(selected)
                for sym in selected:
                    target[sym] = per_slot

        # --- Generate orders ---
        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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
    return sp500_tickers() + ["SPY", "^TYX", "^TNX", "TLT", "IEF"] + DEFENSIVE_SECTORS


UNIVERSE = _universe

NAME = "gen9_longend_slope_sector_rotation"
HYPOTHESIS = (
    "TYX-TNX slope vs 200d MA gates SP500 momentum vs defensive sectors: "
    "slope steepening (above 200d MA) -> top-15 SP500 by 63d momentum; "
    "slope flattening -> top-3 defensive sectors (XLU,XLV,XLP,XLI) by 42d return; "
    "SPY<200d SMA -> TLT+IEF; biweekly rebalance."
)

STRATEGY = LongendSlopeSectorRotation()
