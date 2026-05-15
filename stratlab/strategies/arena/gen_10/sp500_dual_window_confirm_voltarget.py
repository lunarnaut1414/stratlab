"""SP500 dual-window momentum confirmation with portfolio vol-targeting.

Hypothesis (sonnet-10, gen_10):
    Rank SP500 stocks by 126d momentum, but require that the stock ALSO has
    positive 42d return (short-term confirming long-term). This dual-window
    filter eliminates stocks that ranked high on 6-month return but have
    already reversed in the past 2 months — recent reversals that RSI alone
    might miss. Portfolio exposure is vol-targeted to 12% annualized via 30d
    realized portfolio vol. SPY 200d outer bear gate to IEF.

Diversification angle:
  - gen9_sp500_rsi_quality_momentum (OOS 0.88): RSI >= 35 quality screen —
    level-based filter. This uses a RETURN-DIRECTION filter (42d must be
    positive), which catches different reversals than RSI (a stock can have
    RSI 40 but still be in a 42d downtrend).
  - gen9_sp500_voltarget_skipmon (OOS 0.86): skip-month (exclude last 21d),
    no short-window confirmation. This strategy requires POSITIVE 42d return
    (opposite direction — confirms recent strength, not avoids reversal).
  - No leaderboard strategy uses simultaneous dual positive-momentum windows
    (42d AND 126d both positive) combined with portfolio vol-targeting.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_LONG = 126             # 6-month primary momentum
MOM_SHORT = 42             # 2-month confirmation window
SPY_TREND_WINDOW = 200
TOP_K = 15
VOL_TARGET = 0.12          # 12% annualized portfolio vol target
VOL_WINDOW = 30            # 30d realized portfolio vol lookback
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_WINDOW_INV = 21        # per-stock inverse-vol weighting
ANNUALIZATION = 252


class SP500DualWindowConfirmVoltarget(Strategy):
    """SP500 126d momentum with positive-42d confirmation filter;
    portfolio vol-targeting; SPY 200d gate; IEF defensive.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = MOM_LONG + 10
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

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE_MAX
        else:
            need = MOM_LONG + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOM_LONG:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_LONG + 2:
                    continue

                # Long-window momentum (126d)
                p_end = float(col.iloc[-1])
                p_start_long = float(col.iloc[-MOM_LONG])
                if p_start_long <= 0 or not np.isfinite(p_start_long):
                    continue
                ret_long = p_end / p_start_long - 1.0
                if not np.isfinite(ret_long) or ret_long <= 0:
                    continue  # must be positive 126d return

                # Short-window confirmation (42d must also be positive)
                if len(col) < MOM_SHORT + 2:
                    continue
                p_start_short = float(col.iloc[-MOM_SHORT])
                if p_start_short <= 0 or not np.isfinite(p_start_short):
                    continue
                ret_short = p_end / p_start_short - 1.0
                if not np.isfinite(ret_short) or ret_short <= 0:
                    continue  # must confirm with positive 42d return

                # Per-stock inverse-vol weight
                tail = col.values[-(VOL_WINDOW_INV + 1):]
                if len(tail) < VOL_WINDOW_INV + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret_long
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE_MAX
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                # Portfolio vol-targeting
                vol_prices = ctx.closes_window(VOL_WINDOW + 5)
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
                    if annual_vol > 1e-6:
                        scale = VOL_TARGET / annual_vol
                    else:
                        scale = 1.0
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                # Inverse-vol weighted allocation
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["SPY", "IEF"]


UNIVERSE = _universe

NAME = "sp500_dual_window_confirm_voltarget"
HYPOTHESIS = (
    "SP500 top-15 by 126d momentum with dual-window momentum confirmation: stock must also "
    "have positive 42d return (short-term momentum confirming long-term trend); inverse-vol "
    "weighted; portfolio 12pct vol-targeting (30d realized, clip 50-97%); SPY 200d outer gate "
    "to IEF; biweekly rebalance — dual 42d+126d positive confirmation eliminates recent "
    "reversal stocks that still rank high on 6-month return"
)

STRATEGY = SP500DualWindowConfirmVoltarget()
