"""Dividend-Growth vs Tech-Growth ETF Rotation — gen_8 sonnet-9

Hypothesis: Compare VIG (dividend-growth ETF) vs QQQ (tech-growth ETF)
42-day momentum. When QQQ leads hold QQQ at 97%. When VIG leads (dividend-
growth winning = risk appetite shifting to quality/value) hold VIG 60% +
IEF 37%. When both are below their 200d SMA (bear market) hold TLT 97%.
Weekly rebalance.

Rationale: Dividend-growth vs tech-growth relative momentum captures the
growth-vs-quality style rotation cycle. In the 2010-2018 IS window, QQQ
dominates when growth is in favor, while VIG outperforms during uncertainty
(2011, 2015-2016). The rotation into IEF alongside VIG during dividend-
led periods reduces volatility without fully exiting equities. No existing
leaderboard strategy uses VIG as a primary signal or holding.

Differentiation: All existing ETF rotation strategies compare within
equity-bond (SPY/QQQ/TLT) or macro-credit (JNK/HYG) dimensions. The
VIG/QQQ style-momentum dimension is a novel axis — dividend-growth stock
selection vs tech momentum rather than any macro-regime gate.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

MOMENTUM_WINDOW = 42      # ~2 months
TREND_WINDOW = 200        # 200d SMA bear gate
REBALANCE_DAYS = 5        # Weekly
EXPOSURE = 0.97

UNIVERSE = ["VIG", "QQQ", "IEF", "TLT", "SPY"]


class VigQqqDividendGrowthRotation(Strategy):
    """Dividend-growth vs tech-growth ETF rotation using 42d relative momentum."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = TREND_WINDOW + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        # Check all required tickers available
        required = ["VIG", "QQQ", "IEF", "TLT", "SPY"]
        for sym in required:
            if sym not in closes.index or closes[sym] <= 0:
                return []

        prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
        if len(prices_window) < MOMENTUM_WINDOW:
            return []

        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- Compute 42-day momentum for VIG and QQQ ---
        def get_momentum(sym: str) -> float | None:
            if sym not in prices_window.columns:
                return None
            col = prices_window[sym].dropna()
            if len(col) < MOMENTUM_WINDOW:
                return None
            p_end = float(col.iloc[-1])
            p_start = float(col.iloc[-MOMENTUM_WINDOW])
            if p_start <= 0:
                return None
            return p_end / p_start - 1.0

        vig_mom = get_momentum("VIG")
        qqq_mom = get_momentum("QQQ")

        if vig_mom is None or qqq_mom is None:
            return []

        # --- Trend gate: check both VIG and QQQ vs 200d SMA ---
        def above_200d(sym: str) -> bool:
            hist = ctx.history(sym)
            if len(hist) < TREND_WINDOW:
                return False
            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
            price = live.get(sym, 0.0)
            return price > sma

        vig_trend = above_200d("VIG")
        qqq_trend = above_200d("QQQ")

        # --- Determine regime and target ---
        if not vig_trend and not qqq_trend:
            # Bear market: full TLT defensive
            target = {"TLT": EXPOSURE}
        elif qqq_mom >= vig_mom and qqq_trend:
            # QQQ leading: hold QQQ fully
            target = {"QQQ": EXPOSURE}
        elif vig_mom > qqq_mom and vig_trend:
            # VIG (dividend-growth) leading: hold VIG + IEF blend
            target = {"VIG": 0.60, "IEF": 0.37}
        elif qqq_trend:
            # VIG leading but QQQ still in trend, or only QQQ above 200d
            target = {"QQQ": EXPOSURE}
        else:
            # VIG in trend, QQQ not: hold VIG + IEF blend
            target = {"VIG": 0.60, "IEF": 0.37}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym, 0.0)
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


NAME = "vig_qqq_dividend_growth_rotation"
HYPOTHESIS = (
    "Dividend-growth vs tech-growth rotation: use VIG 42d return vs QQQ 42d return; "
    "QQQ leads -> hold QQQ 97%; VIG leads -> hold VIG 60%+IEF 37%; both below 200d SMA -> TLT; "
    "weekly rebalance; dividend-growth signal novel and uncorrelated to VIX/credit/yield gates."
)

STRATEGY = VigQqqDividendGrowthRotation()
