"""CBOE SKEW tail-risk contrarian regime — gen_8 opus-5 (wildcard, attempt 2).

Hypothesis: The CBOE SKEW Index (^SKEW) measures the implied left-tail risk
priced into SP500 options — derived from the relative cost of out-of-the-money
puts vs ATM options. This is a structurally DIFFERENT signal from VIX (which
measures ATM 30-day vol) and from VVIX (vol-of-vol). No strategy on the
leaderboard or in any prior round uses ^SKEW as a primary regime signal —
this is genuine wildcard territory.

Contrarian interpretation:
  - HIGH SKEW (>135 on 21d SMA): Professional option market is pricing
    elevated tail risk → puts expensive, dealers/hedgers are net hedged.
    Historically a "hedged market" condition is RESILIENT (the marginal
    seller has already protected; forced selling is unlikely). Forward
    1-3 month equity returns under high-SKEW regimes have been positive
    on average since 2010, contrary to naive interpretation. ⇒ Risk-on
    (hold QQQ).
  - LOW SKEW (<125 on 21d SMA): Complacent option market, puts cheap,
    no tail-hedging. Historically precedes sharper drawdowns when shocks
    arrive (unhedged market = more forced selling on adverse moves).
    ⇒ Pre-emptive defensive (lean into TLT/IEF).
  - MID-SKEW (125-135): Neutral / SPY-IEF balanced.

Why anti-consensus:
  - ^SKEW NOT used in any prior round (verified via grep).
  - All vol-regime strategies use VIX level/percentile. SKEW captures a
    different facet — the SHAPE of the vol surface (skew), not its level.
  - ^MOVE (rate vol) and ^VVIX (vol-of-vol) have been used in gen_6/gen_7
    but the option-skew dimension is untouched.
  - Contrarian interpretation (HIGH SKEW = risk-on) is itself anti-consensus
    — even if a future agent tried SKEW, they'd likely default to the naive
    "SKEW up = risk-off" framing.

Prohibited check:
  - Not SP500 cross-sectional momentum ✓
  - Not VIX gating ✓ (this is SKEW, not VIX, and the rationale is
    fundamentally different — vol surface shape, not level)
  - Not JNK credit ✓
  - Not yield curve ✓
  - Not seasonal/calendar ✓
  - Not sector rotation ✓
  - Not commodity rotation ✓
  - Not gold vs equity ✓
  - Not PFF preferred ✓
  - Not credit divergence canary ✓

Data:
  - ^SKEW cached from 1990-01-02 → full IS window.
  - SPY, QQQ, TLT, IEF, SHY all cached.

Regime logic (popular_etfs universe, weekly rebalance):
  1. Outer bear gate: SPY < 200d SMA → TLT 60% + SHY 37%.
  2. Compute SKEW 21-day SMA (smooth out daily noise).
  3. SKEW_sma > 135 AND SPY bull: hold QQQ 97% (hedged market, growth-on).
  4. SKEW_sma between 125 and 135: hold SPY 60% + IEF 37%.
  5. SKEW_sma < 125 AND SPY bull: hold SPY 40% + TLT 35% + IEF 22%
     (preemptive defensive — complacent market, unhedged, tail risk).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5
SKEW_SMA = 21
TREND_WINDOW = 200
SKEW_HIGH = 135.0
SKEW_LOW = 125.0
EXPOSURE = 0.97

_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_IEF = "IEF"
_SHY = "SHY"
_SKEW = "^SKEW"

UNIVERSE = "popular_etfs"


class SkewTailRiskContrarian(Strategy):
    """^SKEW 21d SMA contrarian regime: high SKEW = QQQ; low SKEW = preemptive defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        skew_sma: int = SKEW_SMA,
        trend_window: int = TREND_WINDOW,
        skew_high: float = SKEW_HIGH,
        skew_low: float = SKEW_LOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            skew_sma=skew_sma,
            trend_window=trend_window,
            skew_high=skew_high,
            skew_low=skew_low,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.skew_sma = int(skew_sma)
        self.trend_window = int(trend_window)
        self.skew_high = float(skew_high)
        self.skew_low = float(skew_low)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.skew_sma) + 10
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

        # --- SPY 200d SMA bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except Exception:
            return []
        if spy_hist is None or len(spy_hist) < self.trend_window + 5:
            return []
        spy_cl = spy_hist["close"].dropna()
        if len(spy_cl) < self.trend_window:
            return []
        spy_bull = float(spy_cl.iloc[-1]) > float(spy_cl.iloc[-self.trend_window:].mean())

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: TLT 60% + SHY 37%
            for sym, w in [(_TLT, 0.60), (_SHY, 0.37)]:
                if sym in live:
                    target[sym] = w * self.exposure
        else:
            # --- SKEW 21d SMA ---
            skew_sma_val = None
            try:
                skew_hist = ctx.history(_SKEW)
                if skew_hist is not None and len(skew_hist) >= self.skew_sma + 5:
                    skew_cl = skew_hist["close"].dropna()
                    if len(skew_cl) >= self.skew_sma:
                        skew_sma_val = float(skew_cl.iloc[-self.skew_sma:].mean())
            except Exception:
                pass

            if skew_sma_val is None or not np.isfinite(skew_sma_val):
                # Default: neutral
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            elif skew_sma_val > self.skew_high:
                # High SKEW: market is hedged, contrarian risk-on
                if _QQQ in live:
                    target[_QQQ] = self.exposure
                else:
                    for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                        if sym in live:
                            target[sym] = w * self.exposure
            elif skew_sma_val < self.skew_low:
                # Low SKEW: complacent, unhedged, preemptive defensive
                for sym, w in [(_SPY, 0.40), (_TLT, 0.35), (_IEF, 0.22)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                # Mid SKEW (125-135): neutral SPY+IEF
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure

        # --- Execute ---
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


NAME = "opus5_skew_tailrisk_contrarian"
HYPOTHESIS = (
    "CBOE ^SKEW 21d SMA regime contrarian: SKEW>135 (hedged market) hold QQQ "
    "97%; SKEW 125-135 hold SPY 60%+IEF 37%; SKEW<125 (complacent) hold SPY "
    "40%+TLT 35%+IEF 22% preemptive defensive; SPY<200d bear gate hold TLT "
    "60%+SHY 37%; weekly rebalance; popular_etfs universe; option-skew shape "
    "signal absent from leaderboard distinct from VIX level and VVIX vol-of-vol"
)

STRATEGY = SkewTailRiskContrarian()
