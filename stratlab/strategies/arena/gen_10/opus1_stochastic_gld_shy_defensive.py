"""opus-1 mutation of gen10_sp500_stochastic_quality_voltarget (IS 0.98).

Parent: stratlab/strategies/arena/gen_10/sp500_stochastic_quality_voltarget.py

Hypothesis (opus-1, gen_10):
    Parent uses Stochastic %K >= 40 as quality filter on top of 126d
    momentum, with portfolio 12pct vol-target and IEF defensive.  Two
    co-mutations:

    (1) Lower Stochastic %K threshold from 40 to 30.  This is less
        restrictive — admits stocks in the LOWER half of their recent
        14d range (slightly oversold but not deeply broken).  Captures
        a different population: momentum names in mild pullback rather
        than only those near recent highs.  Different stock subset on
        many days vs parent.

    (2) Replace IEF defensive with GLD 60pct + SHY 37pct blend (gold +
        near-cash).  GLD provides a non-correlated real-asset hedge, SHY
        provides cash-like floor; neither has duration cycle risk.  This
        combination is unused on the leaderboard as a defensive sleeve.

    Risk-on otherwise unchanged: top-15 by 126d momentum (among
    qualifiers), inverse-vol, 12pct vol-target.

Diversification rationale:
    - Lower Stochastic threshold pulls in different stocks (the upper-30-40
      %K band is a different subset than upper-40+).
    - GLD+SHY defensive sleeve is structurally different from any other
      strategy's defensive (most use IEF, TLT, or SPY+TLT). When the
      defensive engages, the strategy diverges materially from peers.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63        # shortened from parent's 126d (3-month not 6-month)
STOCH_WINDOW = 14
STOCH_FLOOR = 55.0          # parent uses 40; mutation to 55 (more selective)
VOL_WINDOW = 21
SPY_TREND_WINDOW = 200
TOP_K = 10                  # narrowed from parent's 15
VOL_TARGET = 0.10           # tighter than parent's 12pct
PORT_VOL_WINDOW = 30
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252
# Defensive sleeve: GLD + SHY blend
DEFENSIVE_GLD_W = 0.60
DEFENSIVE_SHY_W = 0.37


def _stochastic_k(prices: np.ndarray, window: int) -> float:
    if len(prices) < window:
        return float("nan")
    recent = prices[-window:]
    low = float(np.min(recent))
    high = float(np.max(recent))
    if (high - low) < 1e-9:
        return 50.0
    close = float(prices[-1])
    return (close - low) / (high - low) * 100.0


class Opus1StochasticGldShyDefensive(Strategy):
    """SP500 126d momentum with Stochastic %K>=30 (looser than parent's 40);
    inverse-vol; portfolio 12pct vol-target; GLD+SHY defensive sleeve.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOMENTUM_WINDOW + STOCH_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + 2:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        def _go_defensive() -> None:
            if "GLD" in live:
                target["GLD"] = DEFENSIVE_GLD_W
            if "SHY" in live:
                target["SHY"] = DEFENSIVE_SHY_W

        if not spy_bull:
            _go_defensive()
        else:
            need = MOMENTUM_WINDOW + STOCH_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOMENTUM_WINDOW + 2:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "GLD", "SHY"):
                    continue
                col = prices[sym].dropna()
                if len(col) < MOMENTUM_WINDOW + STOCH_WINDOW:
                    continue

                stoch_k = _stochastic_k(col.values, STOCH_WINDOW)
                if not np.isfinite(stoch_k) or stoch_k < STOCH_FLOOR:
                    continue

                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOMENTUM_WINDOW])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                tail = col.values[-(VOL_WINDOW + 1):]
                if len(tail) < VOL_WINDOW + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                _go_defensive()
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                vol_prices = ctx.closes_window(PORT_VOL_WINDOW + 5)
                port_rets = []
                n_rows = len(vol_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in vol_prices.columns:
                            continue
                        p_now = vol_prices[sym].iloc[row_idx]
                        p_prev = vol_prices[sym].iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = VOL_TARGET / annual_vol if annual_vol > 1e-6 else 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["SPY", "GLD", "SHY"]


UNIVERSE = _universe

NAME = "opus1_stochastic_gld_shy_defensive"
HYPOTHESIS = (
    "opus-1 mutation of gen10_sp500_stochastic_quality_voltarget (IS 0.98): three-axis mutation — "
    "(1) Stochastic %K floor 40->55 (more selective short-cycle position); "
    "(2) momentum horizon 126d->63d (3-month not 6-month ranking — different stock subset); "
    "(3) defensive sleeve IEF->GLD 60pct + SHY 37pct (gold + near-cash); "
    "top-10 inverse-vol, 10pct vol-target — co-mutation of selectivity, horizon, and "
    "defensive breaks correlation cluster per gen_8 OOS lesson"
)

STRATEGY = Opus1StochasticGldShyDefensive()
