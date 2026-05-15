"""SP500 Skip-Month Momentum with Portfolio Volatility Targeting.

Hypothesis (sonnet-5, gen_9):
    Equal-weight momentum rotation across dividend/income ETFs replaced by:
    Hold top-15 SP500 stocks by 126d-skip-21d momentum, with total portfolio
    exposure dynamically sized to target 12% annualized volatility based on
    30-day realized portfolio return vol. Exposure ranges from 50% (high-vol)
    to 97% (low-vol). SPY 200d SMA gate: below SMA, go to IEF.
    Biweekly rebalance (10 bars).

Diversification angle vs leaderboard:
  - gen8_sp500_skipmon_63sma_momentum (OOS 0.63): uses SPY 63d SMA gate,
    no vol-targeting. This strategy: SPY 200d SMA gate + DYNAMIC EXPOSURE
    via portfolio vol targeting — vol-target changes net equity exposure daily.
  - xsect_12m_invvol_goldencross (curated): uses INVERSE-VOL WEIGHTING of
    individual stocks, not portfolio-level vol targeting. Different mechanism:
    inv-vol changes cross-sectional weights; vol-target changes aggregate exposure.
  - No leaderboard strategy combines 126d-skip-21d momentum + portfolio-level
    vol-target sizing (as opposed to individual stock inv-vol weighting).
  - The vol-target mechanism reduces drawdown in high-vol regimes organically
    without requiring a VIX-level gate that's temporally unstable.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_LOOKBACK = 126          # 6 months
MOM_SKIP = 21               # skip last 1 month (Jegadeesh-Titman)
TREND_WINDOW = 200          # SPY 200d SMA gate
TOP_K = 15                  # number of stocks to hold
VOL_TARGET = 0.12           # 12% annualized portfolio vol target
VOL_WINDOW = 30             # 30d realized portfolio vol lookback
EXPOSURE_MIN = 0.50         # floor exposure
EXPOSURE_MAX = 0.97         # ceiling exposure
ANNUALIZATION = 252         # trading days per year


class Sp500VoltargetSkipmon(Strategy):
    """SP500 126d-skip-21d momentum with portfolio vol-targeting."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LOOKBACK + MOM_SKIP + VOL_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # --- SPY 200d SMA gate ---
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
            # Bear market: IEF defensive
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE_MAX
        else:
            # Bull: select top-K stocks by 126d-skip-21d momentum
            need = MOM_LOOKBACK + MOM_SKIP + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_LOOKBACK + MOM_SKIP:
                    continue
                # Skip-month: p_end = price at -MOM_SKIP, p_start = price at -(MOM_LOOKBACK+MOM_SKIP)
                p_end = float(col.iloc[-MOM_SKIP - 1])
                p_start = float(col.iloc[-(MOM_LOOKBACK + MOM_SKIP)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < TOP_K:
                # Not enough candidates
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE_MAX
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:TOP_K]
                # Equal-weight base allocation (we'll scale by vol target below)
                base_weight = 1.0 / len(ranked)

                # --- Estimate realized portfolio vol over trailing VOL_WINDOW days ---
                # Compute equal-weight portfolio daily log-returns over VOL_WINDOW
                vol_prices = ctx.closes_window(VOL_WINDOW + 5)
                port_rets = []
                n_rows = len(vol_prices)
                for row_idx in range(1, n_rows):
                    row_ret = 0.0
                    count = 0
                    for sym in ranked:
                        if sym not in vol_prices.columns:
                            continue
                        col = vol_prices[sym]
                        p_now = col.iloc[row_idx]
                        p_prev = col.iloc[row_idx - 1]
                        if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                            row_ret += np.log(float(p_now) / float(p_prev))
                            count += 1
                    if count > 0:
                        port_rets.append(row_ret / count)

                if len(port_rets) >= 10:
                    daily_vol = float(np.std(port_rets))
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    if annual_vol > 1e-6:
                        scale = VOL_TARGET / annual_vol
                    else:
                        scale = 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                per_slot = exposure / len(ranked)
                for sym in ranked:
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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "gen9_sp500_voltarget_skipmon"
HYPOTHESIS = (
    "Top-15 SP500 by 126d-skip-21d momentum with portfolio vol-targeting: "
    "exposure = clip(12pct_vol_target / 30d_realized_portfolio_vol, 50%, 97%); "
    "IEF defensive when SPY below 200d SMA; biweekly rebalance."
)

STRATEGY = Sp500VoltargetSkipmon()
