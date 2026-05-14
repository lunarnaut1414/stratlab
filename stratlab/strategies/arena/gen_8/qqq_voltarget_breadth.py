"""QQQ Vol-Targeted with RSP Breadth Overlay — gen_8 sonnet-3

Hypothesis: Hold QQQ sized to 15% annualized vol target (20d realized vol);
boost QQQ exposure to 97% max when RSP/SPY 20d return > 0 (broad breadth
confirms bull regime); reduce to 65% when RSP underperforms SPY (narrow
leadership); TLT when QQQ below 200d SMA; weekly rebalance.

Rationale: Always-invested vol-targeting on QQQ provides steady compounding.
Adding an RSP/SPY breadth overlay dynamically modulates exposure — boosting
when market internals confirm the rally (broad participation), reducing when
rally is narrow (often a warning sign). This is distinct from:
- vix_gated strategies (which use VIX level as gate, not internal breadth)
- realized_vol_carry (which uses SPY percentile vs SPY's own vol history)
- atr_momentum_etf (RSP/SPY breadth driving sector ETF selection, not QQQ sizing)

The vol-target + breadth combination creates a dynamic sizer that reacts to
both market risk level and breadth quality simultaneously.

IS window: 2010-2018 (9 years).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5             # Weekly
VOL_TARGET = 0.15               # 15% annualized vol target
VOL_WINDOW = 20                 # 20d realized vol window
TREND_WINDOW = 200              # QQQ 200d SMA gate
BREADTH_WINDOW = 20             # RSP/SPY 20d return for breadth
HIGH_BREADTH_EXPOSURE = 0.97    # QQQ exposure when RSP beats SPY
LOW_BREADTH_EXPOSURE = 0.65     # QQQ exposure when SPY beats RSP
MIN_EXPOSURE = 0.40             # Never go below 40% QQQ (always-in equity)
MAX_EXPOSURE = 0.97
EXPOSURE_CAP = 0.97


class QqqVolTargetBreadth(Strategy):
    """QQQ vol-targeted always-invested with RSP/SPY breadth-driven exposure modulation."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_target: float = VOL_TARGET,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        high_breadth_exposure: float = HIGH_BREADTH_EXPOSURE,
        low_breadth_exposure: float = LOW_BREADTH_EXPOSURE,
        min_exposure: float = MIN_EXPOSURE,
        max_exposure: float = MAX_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_target=vol_target,
            vol_window=vol_window,
            trend_window=trend_window,
            breadth_window=breadth_window,
            high_breadth_exposure=high_breadth_exposure,
            low_breadth_exposure=low_breadth_exposure,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_target = float(vol_target)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.breadth_window = int(breadth_window)
        self.high_breadth_exposure = float(high_breadth_exposure)
        self.low_breadth_exposure = float(low_breadth_exposure)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.vol_window + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Check QQQ 200d SMA (trend gate)
        qqq_hist = ctx.history("QQQ")
        if len(qqq_hist) < self.trend_window + self.vol_window + 5:
            return []
        qqq_close = qqq_hist["close"].dropna()
        if len(qqq_close) < self.trend_window:
            return []
        qqq_sma200 = float(qqq_close.iloc[-self.trend_window:].mean())
        qqq_price = float(qqq_close.iloc[-1])
        qqq_trending = qqq_price > qqq_sma200

        target: dict[str, float] = {}

        if not qqq_trending:
            # QQQ in downtrend — hold TLT
            if "TLT" in live:
                target["TLT"] = EXPOSURE_CAP
        else:
            # Compute QQQ 20d realized vol for vol-targeting
            if len(qqq_close) < self.vol_window + 2:
                vol_target_exposure = self.high_breadth_exposure
            else:
                tail = qqq_close.iloc[-(self.vol_window + 1):]
                log_rets = np.log(tail.values[1:] / tail.values[:-1])
                rv_daily = float(np.std(log_rets))
                rv_annual = rv_daily * np.sqrt(252)
                if rv_annual > 1e-6:
                    vol_target_exposure = self.vol_target / rv_annual
                else:
                    vol_target_exposure = self.max_exposure
                # Clamp to bounds
                vol_target_exposure = float(np.clip(vol_target_exposure, self.min_exposure, self.max_exposure))

            # RSP/SPY breadth check (20d relative return)
            need = self.breadth_window + 5
            prices_df = ctx.closes_window(need)

            breadth_bullish = True  # Default to bullish if data missing
            if "RSP" in prices_df.columns and "SPY" in prices_df.columns:
                rsp_col = prices_df["RSP"].dropna()
                spy_col = prices_df["SPY"].dropna()
                if len(rsp_col) >= self.breadth_window + 1 and len(spy_col) >= self.breadth_window + 1:
                    rsp_ret = float(rsp_col.iloc[-1]) / float(rsp_col.iloc[-self.breadth_window]) - 1.0
                    spy_ret = float(spy_col.iloc[-1]) / float(spy_col.iloc[-self.breadth_window]) - 1.0
                    breadth_bullish = rsp_ret > spy_ret

            # Breadth-modulate the vol-target exposure
            if breadth_bullish:
                # Broad participation — use min(vol_target_exposure, high_breadth_exposure)
                # but at least as much as vol-target says
                base_exposure = min(vol_target_exposure, self.high_breadth_exposure)
                exposure = max(base_exposure, vol_target_exposure)
                exposure = min(exposure, self.high_breadth_exposure)
            else:
                # Narrow leadership — cap at low_breadth_exposure
                exposure = min(vol_target_exposure, self.low_breadth_exposure)

            exposure = float(np.clip(exposure, self.min_exposure, self.max_exposure))

            # QQQ position
            if "QQQ" in live:
                target["QQQ"] = exposure

            # If not fully invested, hold remaining in TLT
            remaining = EXPOSURE_CAP - exposure
            if remaining > 0.05 and "TLT" in live:
                target["TLT"] = remaining

        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


NAME = "qqq_voltarget_breadth"
HYPOTHESIS = (
    "QQQ-centric vol-targeting with RSP breadth overlay: hold QQQ sized to 15% annualized "
    "vol target (20d realized vol); boost QQQ to max 97% when RSP/SPY 20d return > 0 "
    "(broad breadth confirms); reduce to 65% when RSP underperforms; TLT when QQQ below "
    "200d SMA; weekly rebalance; vol-targeted QQQ with breadth-based exposure modulation"
)

UNIVERSE = ["QQQ", "SPY", "RSP", "TLT", "SHY"]

STRATEGY = QqqVolTargetBreadth()
