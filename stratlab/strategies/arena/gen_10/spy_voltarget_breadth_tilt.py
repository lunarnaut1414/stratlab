"""SPY/QQQ breadth-conditional rotation with vol-targeting.

Hypothesis (sonnet-10, gen_10):
    Breadth-conditional ETF rotation: when RSP outperforms SPY on 42d basis
    (broad market breadth confirmed, equal-weight beating cap-weight), hold
    QQQ at vol-targeted exposure (18% ann target via 21d realized vol of QQQ,
    clip 60-97%). When SPY leads RSP (narrow mega-cap leadership), hold SPY at
    lower vol-target (12% ann). When SPY below 200d SMA, hold TLT.

Rationale:
  - RSP vs SPY return spread is a genuine breadth signal: when equal-weight
    beats cap-weight, it means MOST stocks are rising (breadth confirmed),
    which historically supports momentum continuation into QQQ.
  - When cap-weight leads (narrow rally), SPY at lower exposure is safer.
  - PURE ETF STRATEGY: no individual stock selection → structurally orthogonal
    to all SP500 cross-sectional momentum strategies.
  - Distinct from gen7_realized_vol_carry_spy (SPY 3-tier by realized vol
    percentile): this uses RSP/SPY BREADTH as the primary routing signal, not
    vol regime. Different signal mechanism AND different vehicle switching.

Diversification:
  - No stock selection → avoids SP500 momentum correlation cluster.
  - Weekly rebalance on 3 ETFs generates adequate trades.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_TREND_WINDOW = 200     # outer bear gate
RV_WINDOW = 21             # realized vol for vol-targeting
QQQ_VOL_TARGET = 0.18      # 18% annualized target for QQQ (higher beta vehicle)
SPY_VOL_TARGET = 0.12      # 12% annualized target for SPY (narrow regime)
EXPOSURE_MIN = 0.60
EXPOSURE_MAX = 0.97
ANNUALIZATION = 252
BREADTH_WINDOW = 42        # RSP vs SPY relative return window


class SPYQQQBreadthRotationVoltarget(Strategy):
    """RSP-breadth-conditional QQQ/SPY rotation with vol-targeting; TLT in bear."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(SPY_TREND_WINDOW, BREADTH_WINDOW, RV_WINDOW) + 10
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
            # Bear: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE_MAX
        else:
            # Check RSP vs SPY breadth
            breadth_on = False
            try:
                rsp_hist = ctx.history("RSP")
                rsp_close = rsp_hist["close"].dropna()
                if len(rsp_close) >= BREADTH_WINDOW + 2 and len(spy_close) >= BREADTH_WINDOW + 2:
                    rsp_ret = float(rsp_close.iloc[-1]) / float(rsp_close.iloc[-BREADTH_WINDOW]) - 1.0
                    spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-BREADTH_WINDOW]) - 1.0
                    breadth_on = rsp_ret > spy_ret
            except KeyError:
                pass

            if breadth_on:
                # Breadth confirmed: hold QQQ vol-targeted to 18%
                vehicle = "QQQ"
                vol_target = QQQ_VOL_TARGET
                try:
                    qqq_hist = ctx.history("QQQ")
                    qqq_close = qqq_hist["close"].dropna()
                    hist_for_vol = qqq_close
                except KeyError:
                    vehicle = "SPY"
                    vol_target = SPY_VOL_TARGET
                    hist_for_vol = spy_close
            else:
                # Narrow leadership: hold SPY vol-targeted to 12%
                vehicle = "SPY"
                vol_target = SPY_VOL_TARGET
                hist_for_vol = spy_close

            # Compute vol-targeted exposure
            arr = hist_for_vol.values
            if len(arr) < RV_WINDOW + 1:
                exposure = EXPOSURE_MAX
            else:
                slice_prices = arr[-(RV_WINDOW + 1):]
                logr = np.log(slice_prices[1:] / slice_prices[:-1])
                daily_vol = float(np.std(logr))
                if daily_vol > 1e-6:
                    annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                    scale = vol_target / annual_vol
                    exposure = float(np.clip(scale, EXPOSURE_MIN, EXPOSURE_MAX))
                else:
                    exposure = EXPOSURE_MAX

            if vehicle in closes_now.index:
                target[vehicle] = exposure
            elif "SPY" in closes_now.index:
                target["SPY"] = exposure

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


UNIVERSE = ["SPY", "QQQ", "TLT", "RSP"]

NAME = "spy_voltarget_breadth_tilt"
HYPOTHESIS = (
    "SPY vol-targeting (12% ann via 21d realized vol, clip 50-97%) with QQQ tilt when RSP "
    "outperforms SPY on 42d basis (breadth-confirmed growth regime) and TLT tilt when SPY "
    "below 200d SMA; pure ETF strategy without stock selection; weekly rebalance — mechanism "
    "is vol-target + breadth-tilt on a single asset, orthogonal to all SP500 cross-sectional "
    "momentum strategies on leaderboard"
)

STRATEGY = SPYQQQBreadthRotationVoltarget()
