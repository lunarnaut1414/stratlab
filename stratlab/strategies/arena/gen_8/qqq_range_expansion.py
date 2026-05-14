"""QQQ daily-range expansion momentum timing (gen_8, sonnet-1).

Hypothesis:
    When QQQ's average daily High-Low range (normalized by close) is
    EXPANDING relative to its own 40-day average, it signals increasing
    institutional participation and directional momentum — hold QQQ 97%.
    When range is CONTRACTING (low-volatility consolidation) or SPY is in
    bear territory, rotate to SPY or TLT.

    More specifically:
    - Compute QQQ's 10-day average (H-L)/close (normalized range)
    - Compare to QQQ's 40-day average normalized range
    - If 10d range > 40d range (expanding) AND SPY > 200d SMA AND
      QQQ 20d return > 0 (positive direction):
        -> QQQ 97% (momentum confirmation via range expansion)
    - If SPY > 200d SMA but range contracting:
        -> SPY 97% (reduce tech concentration)
    - SPY < 200d SMA:
        -> TLT 60% + IEF 37%

    Weekly rebalance. Daily-range expansion as a participation/momentum
    signal for QQQ timing is absent from all prior rounds — existing
    strategies use return-based or VIX-based signals, not H-L range.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
RANGE_FAST = 10             # fast window for normalized range
RANGE_SLOW = 40             # slow window for normalized range
DIRECTION_WINDOW = 20       # QQQ 20d return for direction confirmation
SPY_TREND = 200             # SPY 200d SMA bear gate
REBALANCE_EVERY = 5         # weekly
EXPOSURE = 0.97

NAME = "qqq_range_expansion"
HYPOTHESIS = (
    "QQQ daily H-L range expansion as momentum signal: 10d avg range > 40d avg range "
    "AND QQQ 20d positive AND SPY bull -> QQQ 97%; SPY bull but range contracting "
    "-> SPY 97%; SPY bear -> TLT 60%+IEF 37%; weekly rebalance; H-L range "
    "expansion as novel participation signal absent from all prior rounds"
)


class QQQRangeExpansion(Strategy):
    """QQQ H-L range expansion drives QQQ/SPY/TLT allocation."""

    def __init__(
        self,
        range_fast: int = RANGE_FAST,
        range_slow: int = RANGE_SLOW,
        direction_window: int = DIRECTION_WINDOW,
        spy_trend: int = SPY_TREND,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            range_fast=range_fast,
            range_slow=range_slow,
            direction_window=direction_window,
            spy_trend=spy_trend,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.range_fast = int(range_fast)
        self.range_slow = int(range_slow)
        self.direction_window = int(direction_window)
        self.spy_trend = int(spy_trend)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.range_slow, self.spy_trend, self.direction_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- QQQ range computation ---
        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []

        need_range = self.range_slow + 2
        qqq_close = qqq_hist["close"].dropna()
        qqq_high = qqq_hist["high"].dropna()
        qqq_low = qqq_hist["low"].dropna()

        if len(qqq_close) < need_range or len(qqq_high) < need_range or len(qqq_low) < need_range:
            return []

        # Normalized range = (high - low) / close, per bar
        recent_close = qqq_close.iloc[-need_range:]
        recent_high = qqq_high.iloc[-need_range:]
        recent_low = qqq_low.iloc[-need_range:]

        norm_range = (recent_high.values - recent_low.values) / recent_close.values

        range_fast_avg = float(norm_range[-self.range_fast:].mean())
        range_slow_avg = float(norm_range[-self.range_slow:].mean())

        range_expanding = range_fast_avg > range_slow_avg

        # --- QQQ direction confirmation (20d return > 0) ---
        need_dir = self.direction_window + 2
        if len(qqq_close) < need_dir:
            return []
        qqq_dir_ret = float(qqq_close.iloc[-1]) / float(qqq_close.iloc[-(self.direction_window + 1)]) - 1.0
        qqq_positive = qqq_dir_ret > 0

        # --- Regime routing ---
        if not spy_bull:
            target = {"TLT": 0.60, "IEF": 0.37}
        elif range_expanding and qqq_positive:
            target = {"QQQ": self.exposure}
        else:
            target = {"SPY": self.exposure}

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity_val = ctx.portfolio_value(live)
        if equity_val <= 0:
            return []

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Buy/adjust target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity_val * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


STRATEGY = QQQRangeExpansion()
UNIVERSE = ["SPY", "QQQ", "TLT", "IEF"]
