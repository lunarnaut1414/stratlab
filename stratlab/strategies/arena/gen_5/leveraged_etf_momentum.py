"""Leveraged ETF Trend Rider — gen_5 opus-2 (gap_finder)

Hypothesis: Even modest 2x leveraged ETFs amplify returns substantially
when held during persistent bull trends. Use SSO (2x SPY) as primary
risk-on exposure and TQQQ/UPRO/QLD as opportunistic concentration when
trends are STRONG (positive 21d momentum AND positive 63d momentum).
This is a continuous-trend ladder, not a rotation:

Allocation rules at each rebalance (every 21 bars):
  - SPY < 200d SMA → TLT 60% + SHY 35% (bear refuge)
  - SPY > 200d SMA AND VIX 20dMA > 22 → SPY 92% (bull but turbulent)
  - SPY > 200d SMA AND VIX 20dMA <= 22 AND SPY 21d return < 0 → SPY 90%
  - SPY > 200d SMA AND VIX 20dMA <= 22 AND SPY 21d > 0 AND 63d > 0:
      → SSO 60% (modest leverage, the "engaged" state)

Gap addressed: leaderboard has zero leveraged-ETF strategies. SSO data
since 2006-06-21 — full IS coverage with no missing windows.

IS window: 2010-01-01 to 2018-12-31.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SSO", "SPY", "TLT", "SHY", "^VIX"]

_VIX = "^VIX"
_TREND_WINDOW = 200
_VIX_MA_WINDOW = 20
_VIX_THR = 22.0
_FAST_MOM = 21
_SLOW_MOM = 63
_REBALANCE = 21
_SSO_WEIGHT = 0.60     # 2x leveraged → ~1.2x portfolio beta
_SPY_BULL_TURBULENT = 0.92
_SPY_BULL_LOWMOM = 0.90
_TLT_BEAR = 0.60
_SHY_BEAR = 0.35


class LeveragedEtfTrendRider(Strategy):
    """Two-state SSO trend rider with SPY/TLT/SHY tiers."""

    def __init__(
        self,
        trend_window: int = _TREND_WINDOW,
        vix_ma_window: int = _VIX_MA_WINDOW,
        vix_threshold: float = _VIX_THR,
        fast_mom: int = _FAST_MOM,
        slow_mom: int = _SLOW_MOM,
        rebalance: int = _REBALANCE,
        sso_weight: float = _SSO_WEIGHT,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            vix_ma_window=vix_ma_window,
            vix_threshold=vix_threshold,
            fast_mom=fast_mom,
            slow_mom=slow_mom,
            rebalance=rebalance,
            sso_weight=sso_weight,
        )
        self.trend_window = trend_window
        self.vix_ma_window = vix_ma_window
        self.vix_threshold = vix_threshold
        self.fast_mom = fast_mom
        self.slow_mom = slow_mom
        self.rebalance = rebalance
        self.sso_weight = sso_weight

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        # SPY 200d trend
        bullish = False
        spy_21d = float("nan")
        spy_63d = float("nan")
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].astype(float)
                bullish = float(spy_close.iloc[-1]) > float(
                    spy_close.iloc[-self.trend_window:].mean()
                )
                if len(spy_close) >= self.fast_mom + 1:
                    spy_21d = float(
                        spy_close.iloc[-1] / spy_close.iloc[-self.fast_mom] - 1.0
                    )
                if len(spy_close) >= self.slow_mom + 1:
                    spy_63d = float(
                        spy_close.iloc[-1] / spy_close.iloc[-self.slow_mom] - 1.0
                    )
        except (KeyError, Exception):
            pass

        # VIX MA
        vix_calm = False
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= self.vix_ma_window:
                vix_ma_val = float(
                    vix_hist["close"].iloc[-self.vix_ma_window:].mean()
                )
                vix_calm = vix_ma_val < self.vix_threshold
        except (KeyError, Exception):
            pass

        target: dict[str, float] = {}

        if not bullish:
            # Bear refuge
            for sym, w in [("TLT", _TLT_BEAR), ("SHY", _SHY_BEAR)]:
                if sym in closes_now.index:
                    target[sym] = w
        else:
            # Bull regime: SSO when calm + strong momentum, else SPY
            engaged = (
                vix_calm
                and np.isfinite(spy_21d) and spy_21d > 0
                and np.isfinite(spy_63d) and spy_63d > 0
            )
            if engaged and "SSO" in closes_now.index:
                target["SSO"] = self.sso_weight
            elif vix_calm and "SPY" in closes_now.index:
                target["SPY"] = _SPY_BULL_LOWMOM
            elif "SPY" in closes_now.index:
                target["SPY"] = _SPY_BULL_TURBULENT

        # Build orders
        orders: list[Order] = []

        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "leveraged_etf_momentum"
HYPOTHESIS = (
    "Leveraged ETF trend rider: hold SSO (2x SPY) at 60% when SPY>200d AND "
    "VIX20dMA<22 AND 21d+63d momentum both positive (engaged state); SPY 90% "
    "when bull+calm but momentum mixed; SPY 92% when bull but vol elevated; "
    "TLT60%+SHY35% when SPY<200d; monthly rebalance."
)

STRATEGY = LeveragedEtfTrendRider()
