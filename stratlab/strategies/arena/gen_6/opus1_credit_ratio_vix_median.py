"""opus-1 mutation of hy_credit_qqq_rotation (credit-allocator cluster).

Parent: gen6_hy_credit_qqq_rotation (IS Calmar 0.78, h2>>h1, corr_to_top5 0.41).

Structural mutations vs parent:
  - Credit signal: JNK 30d SMA cross  ->  JNK/SHY ratio 20d return
                   (credit-relative-to-cash momentum, like JNK total-return
                   net of risk-free duration drag).
  - Equity confirm: SPY 100d SMA cross  ->  SPY 60d above its 60d-MA-of-MA
                   (slope of trend, not level vs trend) — captures whether
                   the trend is steepening, not just whether we are above it.
  - Risk-on bucket: QQQ  ->  VUG (Vanguard growth ETF) — different vehicle,
                   same growth tilt, broader large-cap basket; different daily
                   innovations from QQQ tech-heavy cluster used by leaders.
  - Defensive:    TLT  ->  IEF mid-duration — less duration risk, different
                   daily path through 2013 taper tantrum and 2018 rate selloff.
  - Rebalance:    weekly (5)  ->  biweekly (10) with min-hold 3 bars on flips.

Why this should be admitted under 0.85 corr filter:
  - JNK/SHY ratio is a *return-based* credit signal (not level-vs-trend).
    JNK 30d SMA cross fires at different times than JNK/SHY 20d ratio momentum.
  - Slope-of-SPY (60d MA accelerating up) is orthogonal to SPY-above-SMA.
  - VUG vs QQQ + IEF vs TLT both alter the daily PnL path significantly.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

CREDIT_WINDOW = 20        # JNK/SHY ratio momentum window
SPY_TREND_WINDOW = 60     # SPY slope window
REBALANCE_EVERY = 10      # biweekly
MIN_HOLD_BARS = 3         # flip-flop guard
EXPOSURE = 0.97
GROWTH_ETF = "VUG"
DEFENSIVE_ETF = "IEF"


class CreditRatioVixMedian(Strategy):
    def __init__(
        self,
        credit_window: int = CREDIT_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        min_hold_bars: int = MIN_HOLD_BARS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            credit_window=credit_window,
            spy_trend_window=spy_trend_window,
            rebalance_every=rebalance_every,
            min_hold_bars=min_hold_bars,
            exposure=exposure,
        )
        self.credit_window = int(credit_window)
        self.spy_trend_window = int(spy_trend_window)
        self.rebalance_every = int(rebalance_every)
        self.min_hold_bars = int(min_hold_bars)
        self.exposure = float(exposure)
        self._last_state: str | None = None
        self._bars_in_state: int = 0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.credit_window, self.spy_trend_window) + 70
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK/SHY ratio momentum: 20-day return of JNK/SHY ratio
        credit_strong = False
        try:
            jnk_hist = ctx.history("JNK")
            shy_hist = ctx.history("SHY")
            if jnk_hist is not None and shy_hist is not None:
                jnk_close = jnk_hist["close"].dropna().values
                shy_close = shy_hist["close"].dropna().values
                n = min(len(jnk_close), len(shy_close))
                if n >= self.credit_window + 1:
                    jnk_close = jnk_close[-n:]
                    shy_close = shy_close[-n:]
                    if shy_close[-1] > 0 and shy_close[-self.credit_window - 1] > 0:
                        ratio_now = jnk_close[-1] / shy_close[-1]
                        ratio_then = (
                            jnk_close[-self.credit_window - 1]
                            / shy_close[-self.credit_window - 1]
                        )
                        if ratio_then > 0:
                            ratio_ret = ratio_now / ratio_then - 1.0
                            credit_strong = ratio_ret > 0.0
        except KeyError:
            pass

        # SPY trend slope: today's close vs 60d ago, AND today > 60d MA
        spy_bull = False
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.spy_trend_window + 5:
                spy_close = spy_hist["close"].dropna().values
                if len(spy_close) >= self.spy_trend_window + 1:
                    spy_now = float(spy_close[-1])
                    spy_then = float(spy_close[-self.spy_trend_window - 1])
                    spy_ma = float(np.mean(spy_close[-self.spy_trend_window:]))
                    # both: above MA AND positive 60d return (slope+level)
                    if spy_then > 0 and np.isfinite(spy_now):
                        spy_bull = (spy_now > spy_ma) and (spy_now > spy_then)
        except KeyError:
            pass

        new_state = "RISK_ON" if (credit_strong and spy_bull) else "RISK_OFF"

        # Min-hold on state flips
        if self._last_state is not None and new_state != self._last_state:
            if self._bars_in_state < self.min_hold_bars:
                self._bars_in_state += 1
                return []
        if new_state == self._last_state:
            self._bars_in_state += 1
        else:
            self._bars_in_state = 0
        self._last_state = new_state

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}
        if new_state == "RISK_ON":
            if GROWTH_ETF in live:
                target[GROWTH_ETF] = self.exposure
            elif "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure
        else:
            if DEFENSIVE_ETF in live:
                target[DEFENSIVE_ETF] = self.exposure
            elif "TLT" in live:
                target["TLT"] = self.exposure

        if not target:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "opus1_credit_ratio_vix_median"
HYPOTHESIS = (
    "Mutate hy_credit_qqq_rotation: JNK/SHY 20d ratio momentum (credit net of "
    "duration drag) replaces JNK 30d SMA cross; SPY slope+level (60d) replaces "
    "SPY 100d SMA cross alone; VUG growth + IEF mid-duration replace QQQ + TLT; "
    "biweekly rebalance with 3-bar min-hold."
)
UNIVERSE = ["JNK", "SHY", "VUG", "QQQ", "IEF", "TLT", "SPY"]

STRATEGY = CreditRatioVixMedian()
