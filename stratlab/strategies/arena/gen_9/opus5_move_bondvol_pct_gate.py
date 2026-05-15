"""MOVE bond-vol 252d percentile regime gate — QQQ vs SPY style allocator.

Hypothesis: ^MOVE (the MOVE Index, bond-market analog of ^VIX measuring
implied vol on US Treasury options) carries orthogonal information to
equity-vol VIX. It captures rate-stress regimes — Fed surprises, debt-
ceiling fights, bond liquidations, taper tantrums — that don't always
coincide with equity-vol spikes.

Key insight: rate-vol regime should modulate the *growth-vs-quality*
equity allocation, not the equity-vs-bond allocation. Duration-long
growth stocks (QQQ) are most sensitive to rate-vol shocks (cash flows
discounted at uncertain rates); broad-market SPY has lower duration
beta. So:

  Low MOVE pct → QQQ (growth thrives when bond-vol is calm)
  High MOVE pct → SPY (broader equity exposure, lower duration risk)

This makes the strategy a STYLE allocator, not a risk-on/risk-off bond
allocator — sidestepping correlation with the 5 long-end-slope and
5 credit-zscore bond strategies on the leaderboard.

Signal tiers (rolled every 10 bars, ~biweekly):
  1. MOVE 252d pct < 50  (rate-calm regime, bottom half):
     Hold QQQ 97% — growth-favorable.
  2. MOVE 252d pct >= 50 (rate-stress regime, top half):
     Hold SPY 97% — broader equity, lower duration risk.
  3. Outer gate: SPY < SPY 200d SMA → TLT 97% (bear-market drawdowns).

Why anti-consensus:
- No strategy in any generation uses ^MOVE. All 9 gen_9 vol-based strategies
  use equity ^VIX, VIX percentile, VVIX, RV-carry, or VRP.
- The brief explicitly tags ^MOVE as "untouched."
- ETF-only (3 tickers traded) — no SP500 cross-sectional layer.
- Style allocator (QQQ vs SPY) rather than asset allocator (equity vs bond)
  — avoids correlation with the bond-tilt cluster.
- Percentile (not absolute) addresses gen_8 SKEW-threshold failure mode.

Hard constraints: allow_short=False, enforce_cash=True, IS only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # ~biweekly
MOVE_PCT_WINDOW = 252       # 1-year rolling percentile
SPLIT_PCT = 50.0            # bottom-half = QQQ, top-half = SPY
SPY_TREND_WINDOW = 200      # outer bear gate
EXPOSURE = 0.97
_MOVE = "^MOVE"


class MoveBondVolPctGate(Strategy):
    """MOVE 252d percentile style allocator: QQQ when bond-vol calm (bottom
    half), SPY when bond-vol stressed (top half); SPY 200d outer bear gate
    to TLT.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        pct_window: int = MOVE_PCT_WINDOW,
        split_pct: float = SPLIT_PCT,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            pct_window=pct_window,
            split_pct=split_pct,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.pct_window = int(pct_window)
        self.split_pct = float(split_pct)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.pct_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 1:
            return []
        spy_sma200 = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        # MOVE 252d percentile signal
        move_pct: float = float("nan")
        try:
            move_hist = ctx.history(_MOVE)
            if move_hist is not None:
                move_close = move_hist["close"].dropna()
                if len(move_close) >= self.pct_window:
                    window = move_close.iloc[-self.pct_window:]
                    latest = float(move_close.iloc[-1])
                    move_pct = float((window < latest).mean() * 100.0)
        except (KeyError, ValueError):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear gate override → TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif not np.isfinite(move_pct):
            # MOVE unavailable → SPY fallback
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        elif move_pct < self.split_pct:
            # Rate-calm → growth (QQQ)
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
            elif "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            # Rate-stress → broad equity (SPY)
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
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


# Narrow ETF-only universe + MOVE signal
UNIVERSE = ["SPY", "QQQ", "TLT", _MOVE]


NAME = "opus5_move_bondvol_pct_gate"
HYPOTHESIS = (
    "MOVE 252d percentile bond-vol regime gate as QQQ/SPY style allocator: "
    "low MOVE pct (<50) holds QQQ (growth thrives in rate-calm), high MOVE pct (>=50) "
    "holds SPY (broader equity, lower duration risk); SPY 200d outer bear gate to TLT; "
    "biweekly — bond-vol regime orthogonal to equity-VIX (no MOVE strategy in any prior gen)"
)

STRATEGY = MoveBondVolPctGate()
