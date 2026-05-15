"""QQQ dynamic position with fast vol-targeting.

Hypothesis: Hold QQQ sized to a 20% annualized vol target using a short
10-day realized vol window (reacts faster to changing conditions than 21d).
When QQQ is below its 63d SMA, hold SHY as defensive. This is a pure
volatility-targeting single-ETF strategy that:
  - Holds ONE ETF (QQQ), not individual stocks — fundamentally different
    from all SP500 stock pickers on the leaderboard
  - Uses a faster vol window (10d) than the existing gen7 realized_vol_carry_spy (21d)
  - Targets QQQ specifically (tech-heavy) rather than SPY
  - Higher vol target (20%) to maintain equity exposure in calm periods

The faster vol window means position sizing responds more quickly to spikes,
reducing drawdowns from sudden vol jumps (e.g. flash crashes, FOMC shocks).

Design:
  - QQQ exposure = clip(vol_target / qqq_10d_ann_vol, min_exp, max_exp)
  - vol_target = 20% (allows up to ~1.0x exposure when vol is 20%, more when lower)
  - min_exp = 0.30 (always hold at least 30% QQQ when in uptrend)
  - max_exp = 0.97 (cap at 97%)
  - Trend gate: QQQ must be above 63d SMA; else hold SHY at 97%
  - Rebalance every 5 bars (weekly)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
VOL_WINDOW = 10           # fast vol lookback
VOL_TARGET = 0.20         # 20% annualized vol target
ANNUAL_FACTOR = 252.0
QQQ_TREND_WINDOW = 63     # QQQ 63d SMA gate
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.30


class QQQFastVolTarget(Strategy):
    """QQQ sized to 20% ann vol target via 10d realized vol; SHY when QQQ below 63d SMA;
    weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        vol_window: int = VOL_WINDOW,
        vol_target: float = VOL_TARGET,
        qqq_trend_window: int = QQQ_TREND_WINDOW,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            vol_window=vol_window,
            vol_target=vol_target,
            qqq_trend_window=qqq_trend_window,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
        )
        self.rebalance_every = int(rebalance_every)
        self.vol_window = int(vol_window)
        self.vol_target = float(vol_target)
        self.qqq_trend_window = int(qqq_trend_window)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.qqq_trend_window, self.vol_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # QQQ history
        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []
        qqq_close = qqq_hist["close"].dropna()
        if len(qqq_close) < max(self.qqq_trend_window, self.vol_window + 1) + 5:
            return []

        qqq_arr = qqq_close.values

        # QQQ 63d SMA trend gate
        if len(qqq_arr) < self.qqq_trend_window:
            return []
        qqq_sma = float(np.mean(qqq_arr[-self.qqq_trend_window:]))
        qqq_bull = float(qqq_arr[-1]) > qqq_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not qqq_bull:
            if "SHY" in closes_now.index:
                target["SHY"] = self.exposure_max
        else:
            # Fast 10d realized vol
            if len(qqq_arr) < self.vol_window + 1:
                target["SHY"] = self.exposure_max
            else:
                tail = qqq_arr[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                ann_vol = rv * (ANNUAL_FACTOR ** 0.5)

                if ann_vol > 1e-6:
                    scale = self.vol_target / ann_vol
                    exposure = float(np.clip(scale, self.exposure_min, self.exposure_max))
                else:
                    exposure = self.exposure_max

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


NAME = "qqq_fast_voltarget"
HYPOTHESIS = (
    "QQQ dynamic vol-targeted position: hold QQQ sized to 20% annualized vol target using 10d "
    "realized vol; reduce to SHY when QQQ 63d SMA bearish; maximum exposure 97%, minimum 30% "
    "in uptrend; rebalance every 5 bars — vol-targeting QQQ with fast 10d window, faster than "
    "gen7 SPY version (21d), different ETF and vol parameters from all leaderboard entries"
)

UNIVERSE = ["QQQ", "SHY"]

STRATEGY = QQQFastVolTarget()
