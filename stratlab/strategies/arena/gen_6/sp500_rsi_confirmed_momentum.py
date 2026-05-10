"""SP500 momentum with RSI strength confirmation.

Hypothesis: Hold top-20 SP500 stocks by 126d momentum (skip 21d) that also
have RSI(14) > 55 (confirming strength, not overbought yet). Inverse-vol
weighted. SPY 200d SMA gate. TLT when bearish. Biweekly rebalance.

The RSI > 55 filter selects stocks where price momentum is confirmed by
short-term strength, removing stocks that have 6-month gains but are
currently weakening. This differs from pure 6-1 month momentum and 52-week
high breakout strategies already on the leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOM_LOOKBACK = 126        # 6 months
MOM_SKIP = 21             # skip last month
RSI_PERIOD = 14
RSI_FLOOR = 55.0          # must be above this to qualify
VOL_WINDOW = 20
TOP_K = 20
TREND_WINDOW = 200
EXPOSURE = 0.97


def _compute_rsi(closes: pd.Series, period: int) -> float:
    """Compute RSI for the most recent bar."""
    if len(closes) < period + 2:
        return 50.0
    deltas = closes.diff().dropna()
    if len(deltas) < period:
        return 50.0
    recent = deltas.iloc[-period:]
    gains = recent[recent > 0].mean() if (recent > 0).any() else 0.0
    losses = -recent[recent < 0].mean() if (recent < 0).any() else 0.0
    if losses == 0:
        return 100.0
    if gains == 0:
        return 0.0
    rs = gains / losses
    return float(100.0 - 100.0 / (1.0 + rs))


class SP500RsiConfirmedMomentum(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        rsi_period: int = RSI_PERIOD,
        rsi_floor: float = RSI_FLOOR,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            rsi_period=rsi_period,
            rsi_floor=rsi_floor,
            vol_window=vol_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.rsi_period = int(rsi_period)
        self.rsi_floor = float(rsi_floor)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + 5
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
            # Bear market: move to TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Bull market: momentum + RSI confirmation
            need = self.mom_lookback + self.mom_skip + self.rsi_period + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                total_needed = self.mom_lookback + self.mom_skip + self.rsi_period + 2
                if len(col) < total_needed:
                    continue

                # Momentum score (6-1 month)
                end_idx = -self.mom_skip
                start_idx = -(self.mom_lookback + self.mom_skip)
                p_end = float(col.iloc[end_idx])
                p_start = float(col.iloc[start_idx])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                mom = p_end / p_start - 1.0

                # RSI confirmation filter
                rsi_series = col.iloc[-(self.rsi_period + 10):]
                rsi_val = _compute_rsi(rsi_series, self.rsi_period)
                if rsi_val < self.rsi_floor:
                    continue  # Skip stocks with weak short-term strength

                # Inverse-vol weight
                vol_tail = col.iloc[-self.vol_window - 1:]
                if len(vol_tail) < self.vol_window + 1:
                    continue
                logr = np.log(vol_tail.values[1:] / vol_tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = mom
                inv_vols[sym] = 1.0 / rv

            if len(scores) < max(5, self.top_k // 4):
                # Too few qualifying stocks - fall back to SPY
                if "SPY" in live:
                    target["SPY"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    if "SPY" in live:
                        target["SPY"] = self.exposure
                else:
                    for sym in ranked:
                        target[sym] = self.exposure * inv_vols[sym] / iv_sum

        orders: list[Order] = []

        # Liquidate positions not in target
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY"]


NAME = "sp500_rsi_confirmed_momentum"
HYPOTHESIS = (
    "SP500 high-RSI momentum: hold top-20 SP500 stocks by 126d momentum "
    "(skip 21d) that have RSI(14)>55 (confirming strength, not overbought yet), "
    "inverse-vol weighted; SPY 200d SMA gate; TLT defensive; biweekly rebalance; "
    "RSI confirmation filter differentiates from pure momentum"
)

UNIVERSE = _universe

STRATEGY = SP500RsiConfirmedMomentum()
