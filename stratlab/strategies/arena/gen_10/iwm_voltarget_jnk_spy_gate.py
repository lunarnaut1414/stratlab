"""IWM small-cap vol-targeting with JNK credit + SPY trend dual gate.

Hypothesis (sonnet-10, gen_10):
    Hold IWM (small-cap ETF) sized to 12% annualized vol target (21d realized
    vol of IWM, exposure clipped 60-97%) when BOTH:
    1. JNK is above its 50d SMA (credit healthy, risk-on)
    2. SPY is above its 200d SMA (equity bull market confirmed)
    Hold TLT when either gate fails. Weekly rebalance.

Rationale:
  - Small caps (IWM) have higher beta to credit conditions than large caps.
    When credit is healthy (JNK above 50d MA), small caps disproportionately
    benefit. The JNK gate captures this premium.
  - IWM outperforms SPY during certain bull-market phases (2010-2013,
    2016-2017) and the credit gate captures these cycles correctly.
  - Vol-targeting is the gen_9 lesson: structurally regime-invariant
    deleveraging mechanism that reduces cal from volatility, not momentum signals.
  - SINGLE ETF (IWM) → no individual stock selection, different return profile
    from all SP500 cross-sectional strategies.

Diversification:
  - IWM small-cap beta is distinct from QQQ large-cap growth beta
    (gen10_qqq_voltarget_spy_gate just accepted).
  - JNK credit gate changes regime timing vs SPY 200d SMA alone, creating
    different entry/exit points than pure SPY-trend strategies.
  - IWM loss-mode behavior is different from SP500 momentum strategies: small
    caps tend to sell off earlier and recover faster than large-cap momentum.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_TREND_WINDOW = 200     # outer equity trend gate
JNK_MA_WINDOW = 50         # JNK credit gate
RV_WINDOW = 21             # IWM realized vol for vol-targeting
IWM_VOL_TARGET = 0.12      # 12% annualized
EXPOSURE_MIN = 0.60
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class IWMVoltargetJNKSPYGate(Strategy):
    """IWM vol-targeted 12%; dual JNK 50d MA + SPY 200d MA gate; TLT defensive."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(SPY_TREND_WINDOW, JNK_MA_WINDOW, RV_WINDOW) + 10
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

        # JNK credit gate
        credit_ok = False
        try:
            jnk_hist = ctx.history("JNK")
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= JNK_MA_WINDOW + 2:
                jnk_sma = float(jnk_close.iloc[-JNK_MA_WINDOW:].mean())
                jnk_price = float(jnk_close.iloc[-1])
                credit_ok = jnk_price > jnk_sma
        except KeyError:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull or not credit_ok:
            # Either gate fails: TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE_MAX
        else:
            # Both gates pass: IWM vol-targeted
            try:
                iwm_hist = ctx.history("IWM")
                iwm_close = iwm_hist["close"].dropna()
            except KeyError:
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE_MAX
                return []

            if len(iwm_close) < RV_WINDOW + 1:
                if "IWM" in closes_now.index:
                    target["IWM"] = EXPOSURE_MAX
            else:
                arr = iwm_close.values
                slice_prices = arr[-(RV_WINDOW + 1):]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))

                if daily_vol > 1e-6:
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = IWM_VOL_TARGET / annual_vol
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                if "IWM" in closes_now.index:
                    target["IWM"] = exposure

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


UNIVERSE = ["SPY", "IWM", "TLT", "JNK"]

NAME = "iwm_voltarget_jnk_spy_gate"
HYPOTHESIS = (
    "IWM small-cap vol-targeting (12% ann via 21d realized vol of IWM, clip 60-97%) when "
    "JNK above 50d SMA (credit healthy) AND SPY above 200d SMA (trend confirmed); hold TLT "
    "when either gate fails; weekly rebalance — small-cap vehicle with JNK credit gate "
    "creates different loss-mode than all large-cap SP500 momentum strategies; single-ETF "
    "vol-targeted similar to accepted qqq_voltarget_spy_gate but IWM has distinct style exposure"
)

STRATEGY = IWMVoltargetJNKSPYGate()
