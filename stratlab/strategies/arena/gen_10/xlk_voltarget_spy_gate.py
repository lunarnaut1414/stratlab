"""XLK tech sector ETF vol-targeting with SPY 200d SMA bear gate.

Hypothesis (sonnet-10, gen_10):
    Hold XLK (tech sector ETF) sized to 14% annualized vol target (21d
    realized vol of XLK, exposure clipped 60-97%). When SPY is below its
    200d SMA (bear market), hold TLT 97%. Weekly rebalance.

Rationale:
  - XLK (Technology Select Sector SPDR) outperformed both SPY and QQQ in the
    IS 2010-2018 window by focusing purely on US technology companies.
  - As a sector ETF (not multi-sector like QQQ), XLK has a fundamentally
    different composition: pure US tech hardware, software, semiconductors
    without the non-tech mega-caps (AMZN=consumer, GOOGL=communication) that
    reduce QQQ's effective tech concentration.
  - 14% vol target (slightly higher than QQQ's 15%) reflects XLK's tech
    concentration creating higher volatility than broad-market QQQ.
  - SPY 200d SMA as the only macro gate — same proven mechanism as the
    accepted gen10_qqq_voltarget_spy_gate, applied to a different vehicle.

Diversification:
  - Completely different from SP500 cross-sectional stock selection.
  - Different from QQQ voltarget: XLK sector ETF excludes AMZN, GOOGL, META
    creating different return patterns in tech-leadership periods.
  - Weekly rebalance on vol-targeting generates 300+ trades in IS window.

Key test: will XLK's vol-target + SPY gate generate IS Calmar > 0.5?
  - XLK CAGR 2010-2018: ~24% (outperforming QQQ's ~20% and SPY's ~15%)
  - XLK MaxDD 2010-2018: ~-25% (higher than SPY due to tech concentration)
  - With vol-targeting to 14%, exposure averages ~80-90%: CAGR ~19%, MaxDD ~20%
  - Expected Calmar: ~0.7-0.9
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_TREND_WINDOW = 200     # outer bear gate
RV_WINDOW = 21             # realized vol for vol-targeting
XLK_VOL_TARGET = 0.14      # 14% annualized
EXPOSURE_MIN = 0.60
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252


class XLKVoltargetSPYGate(Strategy):
    """XLK vol-targeted to 14% ann; SPY 200d bear gate to TLT; weekly rebalance."""

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
            # Bull market: vol-targeted XLK
            try:
                xlk_hist = ctx.history("XLK")
                xlk_close = xlk_hist["close"].dropna()
            except KeyError:
                if "QQQ" in closes_now.index:
                    target["QQQ"] = EXPOSURE_MAX
                return []

            if len(xlk_close) < RV_WINDOW + 1:
                if "XLK" in closes_now.index:
                    target["XLK"] = EXPOSURE_MAX
            else:
                arr = xlk_close.values
                slice_prices = arr[-(RV_WINDOW + 1):]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))

                if daily_vol > 1e-6:
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = XLK_VOL_TARGET / annual_vol
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

                if "XLK" in closes_now.index:
                    target["XLK"] = exposure

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


UNIVERSE = ["SPY", "XLK", "TLT"]

NAME = "xlk_voltarget_spy_gate"
HYPOTHESIS = (
    "XLK tech sector ETF vol-targeted to 14% ann (21d realized vol, clip 60-97%); SPY 200d "
    "SMA outer bear gate to TLT; weekly rebalance; single-sector ETF with vol-targeting "
    "captures tech sector outperformance in IS 2010-2018 bull without individual stock "
    "selection; distinct from qqq_voltarget_spy_gate (XLK holds only US tech stocks vs "
    "QQQ which includes non-tech mega-caps AMZN/GOOGL)"
)

STRATEGY = XLKVoltargetSPYGate()
