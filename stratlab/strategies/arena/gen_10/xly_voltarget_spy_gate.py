"""XLY consumer discretionary ETF vol-targeting with SPY 200d SMA bear gate.

Hypothesis (sonnet-10, gen_10):
    Hold XLY (consumer discretionary sector ETF) sized to 13% annualized vol
    target (21d realized vol of XLY, exposure clipped 60-97%). When SPY is
    below its 200d SMA (bear market), hold TLT 97%. Weekly rebalance.

Rationale:
  - XLY (Consumer Discretionary Select Sector SPDR) captures Amazon, Tesla,
    Home Depot, McDonald's, Nike — the consumer spending and discretionary
    growth story of the 2010-2018 IS window.
  - As a sector ETF, XLY has fundamentally different composition from XLK
    (technology hardware/software/semiconductors) and QQQ (cap-weighted
    tech + non-tech mega-caps). XLY performance is tied to consumer
    confidence and employment rather than tech capex cycles.
  - 13% vol target (between XLK's 14% and SPY's 12%) reflects XLY's
    moderate-high volatility from cyclical consumer exposure.
  - SPY 200d SMA as the only macro gate — same proven mechanism as accepted
    qqq_voltarget_spy_gate and xlk_voltarget_spy_gate.

Diversification:
  - Different sector composition: consumer discretionary vs technology.
  - XLY includes AMZN (dominant constituent), which is excluded from XLK
    and makes XLY's performance tied to e-commerce/retail cycle.
  - Different beta profile: XLY is more sensitive to consumer credit
    conditions and employment, less to earnings-multiple expansion.
  - Weekly rebalance on vol-targeting generates 300+ trades in IS window.

Key test: will XLY's vol-target + SPY gate generate IS Calmar > 0.5?
  - XLY CAGR 2010-2018: ~20% (driven by AMZN's dominant weighting)
  - XLY MaxDD 2010-2018: ~-25% (consumer cyclical drawdowns)
  - With vol-targeting to 13%, exposure averages ~80-90%: CAGR ~16%, MaxDD ~20%
  - Expected Calmar: ~0.6-0.8
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_TREND_WINDOW = 200     # outer bear gate
RV_WINDOW = 21             # realized vol for vol-targeting
XLY_VOL_TARGET = 0.13      # 13% annualized
EXPOSURE_MIN = 0.60
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class XLYVoltargetSPYGate(Strategy):
    """XLY vol-targeted to 13% ann; SPY 200d bear gate to TLT; weekly rebalance."""

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
            # Bull market: vol-targeted XLY
            try:
                xly_hist = ctx.history("XLY")
                xly_close = xly_hist["close"].dropna()
            except KeyError:
                if "SPY" in closes_now.index:
                    target["SPY"] = EXPOSURE_MAX
                return []

            if len(xly_close) < RV_WINDOW + 1:
                if "XLY" in closes_now.index:
                    target["XLY"] = EXPOSURE_MAX
            else:
                arr = xly_close.values
                slice_prices = arr[-(RV_WINDOW + 1):]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))

                if daily_vol > 1e-6:
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = XLY_VOL_TARGET / annual_vol
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                if "XLY" in closes_now.index:
                    target["XLY"] = exposure

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


UNIVERSE = ["SPY", "XLY", "TLT"]

NAME = "xly_voltarget_spy_gate"
HYPOTHESIS = (
    "XLY consumer discretionary sector ETF vol-targeted to 13% ann (21d realized vol, clip 60-97%); "
    "SPY 200d SMA outer bear gate to TLT; weekly rebalance; consumer discretionary captures "
    "Amazon/retail/auto growth in IS 2010-2018 bull; distinct sector composition from XLK (tech) "
    "and QQQ (cap-weighted tech+non-tech) — XLY's performance tied to consumer confidence and "
    "e-commerce cycle rather than tech capex, creating different return timing"
)

STRATEGY = XLYVoltargetSPYGate()
