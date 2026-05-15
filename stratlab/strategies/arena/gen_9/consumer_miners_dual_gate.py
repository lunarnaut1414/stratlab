"""gen_9 sonnet-1 — Consumer Confidence + Miners Dual Gate → SP500 Momentum

Hypothesis: Use two orthogonal risk signals as a composite gate for SP500 momentum:
1. Consumer confidence: XLY vs XLP 42d return (discretionary leads staples = risk-on)
2. Commodity cycle: GDX vs IAU 21d return (miners lead gold = expansion signal)

Score = (1 if XLY>XLP else -1) + (1 if GDX>IAU else -1)
Range: -2 (both defensive), -0 (neutral), +2 (both risk-on)

Tier logic:
- Score = +2 (both risk-on) → top-15 SP500 by 63d momentum above 200d SMA, 97%
- Score = 0 (one each) → top-10 SP500 by 63d momentum, 80%
- Score = -2 (both defensive) → SPY 97% (stay invested, but diversified/simple)
- SPY 200d SMA outer bear gate overrides ALL to TLT 97%

Rationale:
- XLY/XLP consumer confidence signal is an economic leading indicator.
- GDX/IAU miners vs bullion signal captures commodity cycle expansion.
- Combining them creates 3 distinct equity-allocation tiers.
- Routes to SPY when both defensive — avoids bond/gold duration risk that
  creates high corr to bond-rotation strategies on leaderboard.
- SP500 stocks when both risk-on: generates high trade count (10-bar rebalance).
- Pure equity exposure at different concentrations — no bond/gold ETF holdings.

Coverage (all cover IS 2010-2018):
  XLY (1998), XLP (starts 2010-01-04 ≈ IS start), GDX (2006), IAU (2005),
  SPY (1993), TLT (2002)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

CONSUMER_WINDOW = 42    # XLY vs XLP 42d return
MINERS_WINDOW = 21      # GDX vs IAU 21d return
MOMENTUM_WINDOW = 63    # SP500 stock 63d momentum
TREND_WINDOW = 200      # SPY 200d SMA outer bear gate
STOCK_TREND = 200       # per-stock 200d SMA filter
TOP_K_HIGH = 15         # top-15 in highest tier
TOP_K_MID = 10          # top-10 in middle tier
EXP_HIGH = 0.97         # full exposure
EXP_MID = 0.80          # mid exposure
EXP_LOW = 0.97          # SPY base when both defensive
REBALANCE_EVERY = 10    # biweekly

_XLY = "XLY"
_XLP = "XLP"
_GDX = "GDX"
_IAU = "IAU"
_SPY = "SPY"
_TLT = "TLT"


class ConsumerMinersDualGate(Strategy):
    """XLY/XLP consumer + GDX/IAU miners dual-signal tiered SP500 allocation."""

    def __init__(
        self,
        consumer_window: int = CONSUMER_WINDOW,
        miners_window: int = MINERS_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        stock_trend: int = STOCK_TREND,
        top_k_high: int = TOP_K_HIGH,
        top_k_mid: int = TOP_K_MID,
        exp_high: float = EXP_HIGH,
        exp_mid: float = EXP_MID,
        exp_low: float = EXP_LOW,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            consumer_window=consumer_window,
            miners_window=miners_window,
            momentum_window=momentum_window,
            trend_window=trend_window,
            stock_trend=stock_trend,
            top_k_high=top_k_high,
            top_k_mid=top_k_mid,
            exp_high=exp_high,
            exp_mid=exp_mid,
            exp_low=exp_low,
            rebalance_every=rebalance_every,
        )
        self.consumer_window = int(consumer_window)
        self.miners_window = int(miners_window)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.stock_trend = int(stock_trend)
        self.top_k_high = int(top_k_high)
        self.top_k_mid = int(top_k_mid)
        self.exp_high = float(exp_high)
        self.exp_mid = float(exp_mid)
        self.exp_low = float(exp_low)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window, self.consumer_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market → TLT
            if _TLT in live:
                target[_TLT] = self.exp_high
        else:
            # Compute composite score
            score = 0
            need = max(self.consumer_window, self.miners_window) + 5
            prices = ctx.closes_window(need)

            # Consumer signal: XLY vs XLP 42d
            if _XLY in prices.columns and _XLP in prices.columns:
                xly_col = prices[_XLY].dropna()
                xlp_col = prices[_XLP].dropna()
                if len(xly_col) >= self.consumer_window and len(xlp_col) >= self.consumer_window:
                    xly_ret = float(xly_col.iloc[-1] / xly_col.iloc[-self.consumer_window] - 1.0)
                    xlp_ret = float(xlp_col.iloc[-1] / xlp_col.iloc[-self.consumer_window] - 1.0)
                    if np.isfinite(xly_ret) and np.isfinite(xlp_ret):
                        score += 1 if xly_ret > xlp_ret else -1

            # Miners signal: GDX vs IAU 21d
            if _GDX in prices.columns and _IAU in prices.columns:
                gdx_col = prices[_GDX].dropna()
                iau_col = prices[_IAU].dropna()
                if len(gdx_col) >= self.miners_window and len(iau_col) >= self.miners_window:
                    gdx_ret = float(gdx_col.iloc[-1] / gdx_col.iloc[-self.miners_window] - 1.0)
                    iau_ret = float(iau_col.iloc[-1] / iau_col.iloc[-self.miners_window] - 1.0)
                    if np.isfinite(gdx_ret) and np.isfinite(iau_ret):
                        score += 1 if gdx_ret > iau_ret else -1

            # Route based on composite score
            if score <= -2:
                # Both defensive → SPY (stay invested, avoid duration risk)
                if _SPY in live:
                    target[_SPY] = self.exp_low
            elif score == 0:
                # Mixed signals → top-10 SP500 at 80%
                need_mom = self.momentum_window + 5
                mom_prices = ctx.closes_window(need_mom)
                scores_dict: dict[str, float] = {}
                for sym in mom_prices.columns:
                    if sym in (_SPY, _TLT, _XLY, _XLP, _GDX, _IAU):
                        continue
                    col = mom_prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(ret):
                        continue
                    if len(col) < self.stock_trend:
                        continue
                    sma = float(col.iloc[-self.stock_trend:].mean())
                    if float(col.iloc[-1]) <= sma:
                        continue
                    scores_dict[sym] = ret

                if len(scores_dict) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exp_mid
                else:
                    k = min(self.top_k_mid, len(scores_dict))
                    ranked = sorted(scores_dict, key=scores_dict.__getitem__, reverse=True)[:k]
                    per_w = self.exp_mid / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_w
            else:
                # Score >= 1 (at least one risk-on, or both risk-on) → top-15 SP500 at 97%
                need_mom = self.momentum_window + 5
                mom_prices = ctx.closes_window(need_mom)
                scores_dict = {}
                for sym in mom_prices.columns:
                    if sym in (_SPY, _TLT, _XLY, _XLP, _GDX, _IAU):
                        continue
                    col = mom_prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(ret):
                        continue
                    if len(col) < self.stock_trend:
                        continue
                    sma = float(col.iloc[-self.stock_trend:].mean())
                    if float(col.iloc[-1]) <= sma:
                        continue
                    scores_dict[sym] = ret

                if len(scores_dict) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exp_high
                else:
                    k = min(self.top_k_high, len(scores_dict))
                    ranked = sorted(scores_dict, key=scores_dict.__getitem__, reverse=True)[:k]
                    per_w = self.exp_high / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_w

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
    return sp500_tickers() + [_TLT, _SPY, _XLY, _XLP, _GDX, _IAU]


NAME = "consumer_miners_dual_gate"
HYPOTHESIS = (
    "XLY vs XLP 42d consumer confidence + GDX vs IAU 21d miners composite score "
    "(each +1/-1) for SP500 momentum tiers: score=+2 → top-15 SP500 63d momentum 97%; "
    "score=0 → top-10 at 80%; score=-2 → SPY 97%; SPY 200d bear → TLT; "
    "biweekly rebalance; composite consumer+miners gate routes to SP500 stocks or SPY (no bonds)"
)

UNIVERSE = _universe

STRATEGY = ConsumerMinersDualGate()
