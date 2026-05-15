"""QQQ vol-targeting with SPY 200d SMA bear gate.

Hypothesis (sonnet-10, gen_10):
    Hold QQQ sized to 15% annualized vol target (21d realized vol of QQQ,
    exposure clipped 60-97%). When SPY is below its 200d SMA (bear market),
    hold TLT 97%. Weekly rebalance.

Rationale:
  - QQQ systematically outperforms SPY during growth bull markets (2010-2018
    IS window included). vol-targeting provides automatic deleveraging during
    high-vol periods without requiring regime-switching signals.
  - Single-asset (QQQ) strategy eliminates stock-selection correlation entirely.
  - The 15% vol target (vs 12% in SPY strategies) reflects QQQ's higher
    historical vol and allows higher average exposure.
  - SPY 200d SMA as bear gate is the only macro signal — clean, non-overfit.
  - Different from gen5_opus1_qqq_bollinger_vvix_dipbuy (Bollinger+VVIX
    positioning): this uses REALIZED VOL of QQQ directly, not implied vol.
  - Different from gen7_realized_vol_carry_spy: that vol-targets SPY with
    3-tier RV percentile; this vol-targets QQQ with ratio-based scaling.

IS Calmar expectation:
  - QQQ CAGR 2010-2018 ~20% annually, MaxDD ~20%; Calmar ~1.0
  - Vol-targeting reduces effective exposure ~80-90% average → ~18% CAGR
  - SPY gate catches 2010, 2011, 2015-2016, 2018 bear periods.
  - Expected Calmar ~0.6-1.0 depending on max drawdown during gate failures.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_TREND_WINDOW = 200     # outer bear gate
RV_WINDOW = 21             # realized vol lookback
QQQ_VOL_TARGET = 0.15      # 15% annualized vol target for QQQ
EXPOSURE_MIN = 0.60
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class QQQVoltargetSPYGate(Strategy):
    """QQQ vol-targeted to 15% ann; SPY 200d bear gate to TLT; weekly rebalance."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(SPY_TREND_WINDOW, RV_WINDOW) + 10
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
        if len(spy_close) < SPY_TREND_WINDOW + 5:
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
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE_MAX
        else:
            # Bull market: vol-targeted QQQ
            try:
                qqq_hist = ctx.history("QQQ")
                qqq_close = qqq_hist["close"].dropna()
            except KeyError:
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE_MAX
                return []

            if len(qqq_close) < RV_WINDOW + 1:
                if "QQQ" in closes_now.index:
                    target["QQQ"] = EXPOSURE_MAX
            else:
                arr = qqq_close.values
                slice_prices = arr[-(RV_WINDOW + 1):]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))

                if daily_vol > 1e-6:
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = QQQ_VOL_TARGET / annual_vol
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                if "QQQ" in closes_now.index:
                    target["QQQ"] = exposure

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


UNIVERSE = ["SPY", "QQQ", "TLT"]

NAME = "qqq_voltarget_spy_gate"
HYPOTHESIS = (
    "QQQ vol-targeted to 15% ann (21d realized vol of QQQ, clip 60-97%); SPY 200d SMA outer "
    "bear gate to TLT; weekly rebalance; single-asset QQQ strategy where vol-targeting is "
    "the ONLY active mechanism — QQQ outperforms SPY in IS 2010-2018 bull and vol-targeting "
    "prevents tail risks without requiring stock selection or regime-switching signals"
)

STRATEGY = QQQVoltargetSPYGate()
