"""SP500 momentum with equity-sector defensive on realized-vol spikes.

Hypothesis (sonnet-10, gen_10):
    Hold top-15 SP500 stocks by 126d momentum (inverse-vol weighted) when SPY
    21d realized volatility is below the 60th percentile of its trailing 252d
    distribution (calm regime). When realized-vol spikes above the 60th pct,
    rotate to defensive equity sectors XLU+XLP (equal-weight) rather than bonds.
    SPY 200d bear gate overrides to IEF.

Rationale:
  - All leaderboard defensives rotate to TLT/IEF/GLD (bonds/gold). This
    strategy stays in equities during vol spikes by pivoting to low-beta
    defensive sectors, keeping equity risk premium exposure throughout.
  - Realized-vol percentile is self-normalizing unlike VIX absolute levels;
    the 60th pct threshold is calibrated to the IS window's own vol distribution.
  - Mechanism orthogonal to VIX-level gates (VIX is implied, this uses realized)
    and to all bond/gold defensives on the leaderboard.

Diversification angle vs leaderboard:
  - gen9_sp500_voltarget_skipmon: portfolio-level vol exposure scaling. Here
    the vol regime changes the INSTRUMENT (sectors vs stocks), not exposure.
  - gen7_realized_vol_carry_spy: three-tier SPY exposure scaling. Here the
    defensive is XLU+XLP (equity sectors), not reduced SPY.
  - Realized vol percentile form avoids absolute-threshold instability.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_WINDOW = 126           # 6-month momentum
SPY_TREND_WINDOW = 200
RV_WINDOW = 21             # realized vol lookback (SPY)
RV_PERCENTILE_WINDOW = 252 # distribution lookback for percentile
RV_CALM_PCT = 60           # below 60th pct = calm regime
TOP_K = 15
EXPOSURE = 0.97
VOL_WINDOW_INV = 21        # per-stock inverse-vol weighting


class SP500EquityVolRegimeSectors(Strategy):
    """SP500 126d momentum; defensive = XLU+XLP when SPY realized-vol high;
    SPY 200d bear gate to IEF; inverse-vol weighted; biweekly rebalance.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOM_WINDOW, RV_PERCENTILE_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + RV_PERCENTILE_WINDOW + 5:
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

        if not spy_bull:
            # Bear market outer gate: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE
        else:
            # Compute SPY 21d realized vol and its 252d percentile rank
            spy_returns = spy_close.values
            if len(spy_returns) < RV_PERCENTILE_WINDOW + RV_WINDOW + 5:
                return []

            # Compute rolling 21d realized vol over trailing 252d window
            rv_series = []
            for i in range(RV_PERCENTILE_WINDOW):
                idx_end = len(spy_returns) - RV_PERCENTILE_WINDOW + i + 1
                idx_start = idx_end - RV_WINDOW - 1
                if idx_start < 0:
                    continue
                slice_prices = spy_returns[idx_start:idx_end]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))
                annual_vol = daily_vol * np.sqrt(252)
                rv_series.append(annual_vol)

            if len(rv_series) < 20:
                return []

            current_rv = rv_series[-1]
            pct_rank = float(np.mean(np.array(rv_series[:-1]) < current_rv)) * 100.0

            calm_regime = pct_rank < RV_CALM_PCT

            if not calm_regime:
                # High vol: rotate to defensive equity sectors
                xlu_available = "XLU" in closes_now.index
                xlp_available = "XLP" in closes_now.index
                if xlu_available and xlp_available:
                    target["XLU"] = EXPOSURE / 2
                    target["XLP"] = EXPOSURE / 2
                elif xlu_available:
                    target["XLU"] = EXPOSURE
                elif xlp_available:
                    target["XLP"] = EXPOSURE
                else:
                    if "IEF" in closes_now.index:
                        target["IEF"] = EXPOSURE
            else:
                # Calm regime: select top-K SP500 stocks by 126d momentum
                need = MOM_WINDOW + 5
                prices = ctx.closes_window(need)
                if len(prices) < MOM_WINDOW:
                    return []

                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < MOM_WINDOW + 2:
                        continue

                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-MOM_WINDOW])
                    if p_start <= 0 or not np.isfinite(p_start):
                        continue
                    ret = p_end / p_start - 1.0
                    if not np.isfinite(ret):
                        continue

                    # Per-stock inverse-vol weight
                    tail = col.values[-(VOL_WINDOW_INV + 1):]
                    if len(tail) < VOL_WINDOW_INV + 1:
                        continue
                    logr = np.log(tail[1:] / tail[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    scores[sym] = ret
                    inv_vols[sym] = 1.0 / rv

                if len(scores) < 5:
                    if "IEF" in closes_now.index:
                        target["IEF"] = EXPOSURE
                else:
                    k = min(TOP_K, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        return []
                    for sym in ranked:
                        target[sym] = EXPOSURE * inv_vols[sym] / iv_sum

        # Build orders
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
    return sp500_tickers() + ["SPY", "IEF", "XLU", "XLP"]


UNIVERSE = _universe

NAME = "sp500_equity_vol_regime_sectors"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum staying within equity when SPY 21d realized-vol is below "
    "60th percentile of trailing 252d distribution (calm regime); when vol above 60th pct "
    "rotate to low-vol defensive sectors XLU+XLP equal-weight instead of bonds; SPY 200d "
    "bear gate to IEF; inverse-vol stock weighting; biweekly rebalance — keeps equity "
    "exposure in defensive sectors during vol spikes rather than rotating to bonds, mechanism "
    "orthogonal to all bond-based defensive pivots on leaderboard"
)

STRATEGY = SP500EquityVolRegimeSectors()
