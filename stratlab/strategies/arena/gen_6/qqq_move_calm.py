"""QQQ with ^MOVE bond-vol calm gate — gen_6 sonnet-7

Hypothesis: Hold QQQ 97% when ^MOVE is below its 80d MA (bond volatility
is calm, indicating risk-on global rates environment) AND SPY is above 150d
SMA (no US bear market). Hold TLT 97% when SPY<150d SMA. Hold SPY 97% when
^MOVE is elevated (>80d MA) but SPY still in uptrend (avoid growth overweight
when bond vol is elevated). Rebalance weekly.

Rationale:
  ^MOVE index measures US Treasury option-implied volatility. Low MOVE =
  calm rates = good for growth/technology stocks (longer-duration assets
  benefit from stable rates). When bond vol spikes (MOVE > MA), growth
  stocks underperform because rising rate uncertainty discounts long-duration
  cash flows. Using MOVE as primary equity selection signal (QQQ vs SPY) is
  distinct from using MOVE as a leverage trigger (as in gen5_qqq_base_rare_tqqq).

  Distinct from existing leaderboard:
  - ^MOVE as primary QQQ-vs-SPY selector (not leverage trigger)
  - Three-state: QQQ/SPY/TLT based on bond vol + equity trend
  - Same MOVE signal as gen5_opus1_etf_move_factor_rotation but simpler 2-ETF
    implementation vs that strategy's complex factor-ETF rotation
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
MOVE_MA = 80            # 80d MA of ^MOVE
TREND_WINDOW = 150      # SPY 150d SMA for bear market gate
EXPOSURE = 0.97

_MOVE = "^MOVE"


class QQQMOVECalm(Strategy):
    """QQQ when MOVE calm, SPY when MOVE elevated (but bull), TLT in bear."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        move_ma: int = MOVE_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            move_ma=move_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.move_ma = int(move_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.move_ma, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY bear market gate
        spy_bull = True
        try:
            spy_hist = ctx.history("SPY")
            if len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna().values
                spy_sma = float(np.mean(spy_close[-self.trend_window:]))
                spy_bull = float(spy_close[-1]) > spy_sma
        except Exception:
            pass

        # ^MOVE bond vol signal
        move_calm = True   # default: assume calm if no data
        try:
            move_hist = ctx.history(_MOVE)
            if len(move_hist) >= self.move_ma + 2:
                move_close = move_hist["close"].dropna().values
                move_current = float(move_close[-1])
                move_ma_val = float(np.mean(move_close[-self.move_ma:]))
                move_calm = move_current < move_ma_val
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif move_calm:
            # Bond vol calm: hold QQQ (growth)
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # Bond vol elevated but equity bull: hold SPY (less rate-sensitive)
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

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


NAME = "qqq_move_calm"
HYPOTHESIS = (
    "QQQ with ^MOVE bond-vol calm gate: hold QQQ 97% when ^MOVE < 80d MA (calm rates) "
    "AND SPY > 150d SMA; hold SPY 97% when MOVE elevated but equity bull; TLT when SPY bear; "
    "weekly rebalance; MOVE as QQQ vs SPY selector (distinct from MOVE leverage triggers)"
)
UNIVERSE = ["QQQ", "SPY", "TLT", _MOVE]
STRATEGY = QQQMOVECalm()
