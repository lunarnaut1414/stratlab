"""QQQ volatility-targeted always-invested strategy.

Hypothesis: QQQ (Nasdaq-100 ETF) compounds at higher returns than SPY but
with higher volatility. By dynamically sizing the QQQ position to target
a constant annualized volatility of 15%, we capture QQQ's alpha while
smoothing the risk profile. Scale to SHY when QQQ is in a downtrend.

Signal:
  - Compute 20-day realized annualized volatility of QQQ
  - Target exposure = target_vol / realized_vol (capped at 0.97)
  - Trend gate: when QQQ price < 60d SMA, reduce to min exposure (30%) and
    route remainder to SHY (avoid major drawdowns)
  - Rebalance when current exposure drifts >3% from target

Rationale: Vol-targeting reduces position size during high-vol regimes
(typically correlates with drawdowns) and could increase size during
low-vol (but allow_short=False and 97% cap means max 1x). The QQQ 60d SMA
trend gate prevents being positioned large in a clear downtrend.

Diversification vs gen5_spy_vol_target_trend (already on leaderboard):
  - Uses QQQ instead of SPY (higher growth, higher vol, more tech concentration)
  - Uses 60d SMA instead of 200d SMA trend gate (shorter, faster response)
  - Different defensive bucket: SHY (cash) vs SHY (same but different sizing)
  - Daily-corr path differs because QQQ > SPY in tech bull, less in defensive periods
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

VOL_WINDOW = 20           # 20d realized vol estimate
TARGET_VOL = 0.15         # 15% annualized vol target
TREND_WINDOW = 60         # QQQ 60d SMA trend gate
MAX_EXPOSURE = 0.97
MIN_EXPOSURE = 0.30       # floor when in QQQ downtrend
REBALANCE = 5             # weekly
DRIFT_THRESHOLD = 0.03    # 3% drift triggers rebalance
TRADING_DAYS = 252


class QQQVolTarget(Strategy):
    """QQQ with 15% vol-target sizing and 60d SMA trend gate; SHY remainder."""

    def __init__(
        self,
        vol_window: int = VOL_WINDOW,
        target_vol: float = TARGET_VOL,
        trend_window: int = TREND_WINDOW,
        max_exposure: float = MAX_EXPOSURE,
        min_exposure: float = MIN_EXPOSURE,
        rebalance: int = REBALANCE,
        drift_threshold: float = DRIFT_THRESHOLD,
    ) -> None:
        super().__init__(
            vol_window=vol_window,
            target_vol=target_vol,
            trend_window=trend_window,
            max_exposure=max_exposure,
            min_exposure=min_exposure,
            rebalance=rebalance,
            drift_threshold=drift_threshold,
        )
        self.vol_window = int(vol_window)
        self.target_vol = float(target_vol)
        self.trend_window = int(trend_window)
        self.max_exposure = float(max_exposure)
        self.min_exposure = float(min_exposure)
        self.rebalance = int(rebalance)
        self.drift_threshold = float(drift_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.vol_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []

        # Read QQQ history
        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []
        if qqq_hist is None or len(qqq_hist) < self.trend_window + 5:
            return []
        qqq_close = qqq_hist["close"].dropna()
        if len(qqq_close) < self.vol_window + 1:
            return []

        # Compute 20d realized volatility (annualized)
        log_rets = np.log(qqq_close.values[-self.vol_window - 1:][1:] /
                          qqq_close.values[-self.vol_window - 1:][:-1])
        daily_vol = float(np.std(log_rets))
        ann_vol = daily_vol * np.sqrt(TRADING_DAYS)
        if ann_vol <= 0 or not np.isfinite(ann_vol):
            return []

        # Compute QQQ 60d SMA trend
        if len(qqq_close) >= self.trend_window:
            qqq_sma = float(qqq_close.iloc[-self.trend_window:].mean())
            qqq_bull = float(qqq_close.iloc[-1]) > qqq_sma
        else:
            qqq_bull = True

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Calculate target QQQ exposure
        if not qqq_bull:
            # Downtrend: minimum exposure
            qqq_weight = self.min_exposure
        else:
            # Vol-targeting: scale to hit target_vol
            raw_exposure = self.target_vol / ann_vol
            qqq_weight = min(self.max_exposure, max(self.min_exposure, raw_exposure))

        # Remainder goes to SHY
        shy_weight = max(0.0, self.max_exposure - qqq_weight) * 0.95  # slight underweight to leave buffer

        # Check if we need to rebalance (weekly or drift)
        rebalance_due = (ctx.idx % self.rebalance == 0)

        if not rebalance_due:
            # Check drift
            qqq_price = live.get("QQQ", 0)
            if qqq_price > 0:
                qqq_pos = ctx.position("QQQ").size
                current_qqq_exposure = qqq_pos * qqq_price / equity
                if abs(current_qqq_exposure - qqq_weight) < self.drift_threshold:
                    return []

        target: dict[str, float] = {}
        if "QQQ" in closes_now.index:
            target["QQQ"] = qqq_weight
        if shy_weight > 0.01 and "SHY" in closes_now.index:
            target["SHY"] = shy_weight

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "qqq_vol_target"
HYPOTHESIS = (
    "QQQ volatility-targeted always-invested: hold QQQ sized to 15% annualized vol "
    "target using 20d realized vol; scale down to SHY when QQQ 60d SMA bearish "
    "(price below SMA); max exposure 97%; rebalance every 5 bars with 3% drift "
    "threshold; targets steady QQQ compounding with dynamic vol-sizing"
)

UNIVERSE = ["QQQ", "SHY"]

STRATEGY = QQQVolTarget()
