"""SPY vol-target with IEF complement and breadth quality gate.

Hypothesis (sonnet-3, gen_10):
    Hold SPY sized to target 10% annualized portfolio volatility (using 21d
    realized vol), with the remainder in IEF as a vol-complement. When SPY
    is below its 200d SMA, fully rotate to IEF. Additionally, when RSP
    (equal-weight SP500) 21d return falls below SPY 21d return (narrow
    leadership, breadth deteriorating), reduce SPY exposure by 50% as a
    quality gate.

    Rationale:
      - A pure SPY vol-targeting strategy is structurally different from all
        cross-sectional momentum strategies: it holds ONE asset (SPY) not
        a portfolio of 15 stocks. The risk profile is completely different.
      - Vol-targeting without any VIX-level gate is regime-invariant: when
        markets are calm, full vol-target exposure; when volatile, deleverage.
      - The IEF complement provides a natural bond allocation instead of going
        to cash — better risk-adjusted returns when equities are reduced.
      - The RSP breadth quality gate is a secondary signal that reduces exposure
        in narrow-leadership regimes (mega-cap concentration).
      - Mechanism is completely orthogonal to all SP500 cross-sectional momentum
        strategies: no stock ranking, no quality filters, no 126d momentum.

    Design:
      - Compute SPY 21d realized vol daily.
      - Target SPY exposure = clip(10% / ann_vol, 30%, 90%).
      - Remainder = IEF at (1 - SPY_weight).
      - RSP breadth gate: when RSP 21d return < SPY 21d return, halve SPY weight.
      - SPY 200d SMA bear gate: 100% IEF when SPY below 200d SMA.
      - Rebalance every 5 bars (weekly).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
SPY_VOL_WINDOW = 21        # realized vol lookback for SPY sizing
BREADTH_WINDOW = 21        # RSP vs SPY momentum comparison window
SPY_TREND_WINDOW = 200     # SPY 200d SMA outer gate
SPY_TARGET_VOL = 0.10      # 10% annualized SPY vol target
SPY_MIN_WEIGHT = 0.30      # floor SPY weight
SPY_MAX_WEIGHT = 0.90      # ceiling SPY weight (leave room for IEF)
ANNUALIZATION = 252


class SpyVoltargetIefComplement(Strategy):
    """SPY vol-target with IEF complement; RSP breadth quality gate; SPY 200d SMA
    outer gate; weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        spy_vol_window: int = SPY_VOL_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        spy_target_vol: float = SPY_TARGET_VOL,
        spy_min_weight: float = SPY_MIN_WEIGHT,
        spy_max_weight: float = SPY_MAX_WEIGHT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            spy_vol_window=spy_vol_window,
            breadth_window=breadth_window,
            spy_trend_window=spy_trend_window,
            spy_target_vol=spy_target_vol,
            spy_min_weight=spy_min_weight,
            spy_max_weight=spy_max_weight,
        )
        self.rebalance_every = int(rebalance_every)
        self.spy_vol_window = int(spy_vol_window)
        self.breadth_window = int(breadth_window)
        self.spy_trend_window = int(spy_trend_window)
        self.spy_target_vol = float(spy_target_vol)
        self.spy_min_weight = float(spy_min_weight)
        self.spy_max_weight = float(spy_max_weight)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + self.spy_vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get SPY history
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []

        # SPY 200d SMA bear gate
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
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
            # Bear: 100% IEF
            target["IEF"] = 0.97
        else:
            # --- SPY realized vol targeting ---
            if len(spy_close) < self.spy_vol_window + 2:
                return []
            spy_tail = spy_close.values[-(self.spy_vol_window + 1):]
            spy_daily_rets = np.log(spy_tail[1:] / spy_tail[:-1])
            spy_daily_vol = float(np.std(spy_daily_rets))
            spy_annual_vol = spy_daily_vol * np.sqrt(ANNUALIZATION)

            if spy_annual_vol > 1e-6:
                spy_weight = self.spy_target_vol / spy_annual_vol
            else:
                spy_weight = self.spy_max_weight

            spy_weight = float(np.clip(spy_weight, self.spy_min_weight, self.spy_max_weight))

            # --- RSP breadth quality gate ---
            # When RSP 21d return < SPY 21d return (narrow leadership), halve SPY weight
            try:
                rsp_hist = ctx.history("RSP")
                rsp_close = rsp_hist["close"].dropna()
                if len(rsp_close) >= self.breadth_window + 2:
                    rsp_ret = float(rsp_close.iloc[-1]) / float(rsp_close.iloc[-self.breadth_window]) - 1.0
                    spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-self.breadth_window]) - 1.0
                    if rsp_ret < spy_ret:
                        # Narrow leadership: reduce SPY exposure
                        spy_weight *= 0.5
                        spy_weight = max(spy_weight, self.spy_min_weight)
            except (KeyError, IndexError):
                pass  # RSP not available — skip breadth gate

            ief_weight = max(0.97 - spy_weight, 0.0)

            target["SPY"] = spy_weight
            if ief_weight > 0.01:
                target["IEF"] = ief_weight

        # --- Build orders ---
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


UNIVERSE = ["SPY", "IEF", "RSP"]

NAME = "spy_voltarget_ief_complement"
HYPOTHESIS = (
    "SPY vol-targeted (10% ann target via 21d realized vol, clip 30-90%) with IEF complement "
    "(remainder of allocation); RSP breadth gate: halve SPY when RSP 21d return < SPY 21d return "
    "(narrow leadership); SPY 200d SMA bear gate to 100% IEF; weekly rebalance — pure SPY vol-targeting "
    "with bond complement is mechanistically orthogonal to all SP500 cross-sectional momentum strategies"
)

STRATEGY = SpyVoltargetIefComplement()
