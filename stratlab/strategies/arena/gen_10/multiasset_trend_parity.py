"""Multi-asset trend-following risk parity with QQQ equity tilt.

Hypothesis (sonnet-10, gen_10):
    Core: hold SPY (above 200d SMA), TLT (above 90d SMA), GLD (above 180d
    SMA), each sized inversely by 20d realized vol. SHY for out-of-trend
    assets. TILT: when SPY is in trend AND RSP outperforms SPY on 42d basis
    (breadth confirmed), replace SPY with QQQ in the risk parity allocation
    (same inverse-vol weight, different vehicle). Weekly rebalance.

Rationale:
  - Risk parity alone gives low IS Calmar in 2010-2018 equity bull (bonds/gold
    drag). The QQQ substitution adds equity upside when breadth confirms the
    rally without changing the risk parity STRUCTURE.
  - When breadth is broad (RSP > SPY), QQQ captures tech/growth leadership;
    when narrow (SPY leads), SPY at RP-weight is appropriate.
  - Always-invested (some position always held) avoids the missed-trades problem.
  - NEVER selects individual stocks → structurally orthogonal to all SP500
    cross-sectional momentum strategies. The correlation comes from shared SPY
    exposure, not stock selection.

Different from:
  - gen6_rp_credit_tilt: always-invested SPY+TLT+GLD, JNK credit overlay;
    no trend gates, no QQQ substitution.
  - gen5_risk_parity_spy_tlt_gld: fixed inverse-vol parity, no trend gates,
    no breadth tilt.
  - spy_voltarget_breadth_tilt (gen10 sibling): pure SPY/QQQ rotation,
    no multi-asset parity, no TLT/GLD.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
VOL_WINDOW = 20            # realized vol lookback for risk parity weights
ANNUALIZATION = 252
BREADTH_WINDOW = 42        # RSP vs SPY relative return for QQQ tilt

# Asset-specific trend thresholds
TREND_CONFIG = {
    "SPY": 200,   # 200d SMA for equity
    "TLT": 90,    # 90d SMA for long bonds
    "GLD": 180,   # 180d SMA for gold
}
CASH_ETF = "SHY"
EXPOSURE = 0.97


class MultiAssetTrendParityQQQTilt(Strategy):
    """SPY+TLT+GLD trend-gated risk parity with QQQ tilt on RSP breadth."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(max(TREND_CONFIG.values()), BREADTH_WINDOW) + VOL_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Check RSP vs SPY breadth for QQQ tilt decision
        breadth_on = False
        try:
            spy_hist = ctx.history("SPY")
            spy_close = spy_hist["close"].dropna()
            rsp_hist = ctx.history("RSP")
            rsp_close = rsp_hist["close"].dropna()
            if (len(spy_close) >= BREADTH_WINDOW + 2 and
                    len(rsp_close) >= BREADTH_WINDOW + 2):
                rsp_ret = float(rsp_close.iloc[-1]) / float(rsp_close.iloc[-BREADTH_WINDOW]) - 1.0
                spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-BREADTH_WINDOW]) - 1.0
                breadth_on = rsp_ret > spy_ret
        except KeyError:
            pass

        # Determine which assets are in trend and compute their realized vol
        active_assets: dict[str, float] = {}  # sym -> inv-vol weight

        for sym, trend_window in TREND_CONFIG.items():
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            close = hist["close"].dropna()
            if len(close) < trend_window + VOL_WINDOW + 2:
                continue

            # Trend gate: price above SMA
            sma = float(close.iloc[-trend_window:].mean())
            price = float(close.iloc[-1])
            if price <= sma:
                continue  # Below trend — SHY instead

            # 20d realized vol for risk parity
            slice_prices = close.values[-(VOL_WINDOW + 1):]
            if len(slice_prices) < VOL_WINDOW + 1:
                continue
            logr = np.log(slice_prices[1:] / slice_prices[:-1])
            daily_vol = float(np.std(logr))
            if daily_vol <= 1e-6 or not np.isfinite(daily_vol):
                continue

            # Substitute QQQ for SPY when breadth is broad
            hold_sym = sym
            if sym == "SPY" and breadth_on and "QQQ" in closes_now.index:
                hold_sym = "QQQ"
                # Recompute vol for QQQ
                try:
                    qqq_hist = ctx.history("QQQ")
                    qqq_close = qqq_hist["close"].dropna()
                    if len(qqq_close) >= VOL_WINDOW + 1:
                        qqq_slice = qqq_close.values[-(VOL_WINDOW + 1):]
                        qqq_logr = np.log(qqq_slice[1:] / qqq_slice[:-1])
                        qqq_vol = float(np.std(qqq_logr))
                        if qqq_vol > 1e-6 and np.isfinite(qqq_vol):
                            daily_vol = qqq_vol
                except KeyError:
                    hold_sym = "SPY"

            active_assets[hold_sym] = 1.0 / daily_vol

        target: dict[str, float] = {}

        if not active_assets:
            if CASH_ETF in closes_now.index:
                target[CASH_ETF] = EXPOSURE
        else:
            iv_sum = sum(active_assets.values())
            n_inactive = len(TREND_CONFIG) - len(active_assets)

            if n_inactive > 0 and CASH_ETF in closes_now.index:
                cash_fraction = n_inactive / len(TREND_CONFIG)
                equity_fraction = 1.0 - cash_fraction
                cash_weight = EXPOSURE * cash_fraction
                equity_weight = EXPOSURE * equity_fraction
                target[CASH_ETF] = cash_weight
                for sym, inv_vol in active_assets.items():
                    target[sym] = equity_weight * inv_vol / iv_sum
            else:
                for sym, inv_vol in active_assets.items():
                    target[sym] = EXPOSURE * inv_vol / iv_sum

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


UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "SHY", "RSP"]

NAME = "multiasset_trend_parity"
HYPOTHESIS = (
    "Multi-asset trend-following risk parity: hold SPY when above 200d SMA + TLT when above "
    "90d SMA + GLD when above 180d SMA, each sized inversely by 20d realized volatility "
    "(risk budget allocation); SHY for any asset below its trend threshold; when RSP "
    "outperforms SPY on 42d basis (broad breadth), substitute QQQ for SPY in the RP "
    "allocation; weekly rebalance; always-invested but trend-gated multi-asset parity is "
    "structurally different from stock-selection momentum — avoids the SP500 momentum "
    "correlation cluster entirely"
)

STRATEGY = MultiAssetTrendParityQQQTilt()
