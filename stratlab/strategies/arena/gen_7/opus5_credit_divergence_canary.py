"""Equity-credit divergence canary — gen_7 opus-5 wildcard #2.

Hypothesis
----------
Most credit-gated strategies use credit as a regime filter that's "ON or OFF":
JNK above its MA -> equity, JNK below its MA -> bonds. That spends a lot of time
in bonds during normal credit consolidations and drags IS returns.

This wildcard inverts the framing: credit is a CANARY, not a regime gate.
The default position is offensive (QQQ 97%). We only de-risk when SPY and JNK
*disagree* — specifically when SPY 21d return is positive (equity rallying)
AND JNK 21d return is negative (credit selling off). Historically this divergence
has preceded most major equity drawdowns (mid-2007, late-2014, late-2015,
early-2018) — when stocks keep rising while credit canary stops singing, it's
a high-conviction warning.

Decision rule (weekly rebalance, allow_short=False):
  IF SPY < 200d SMA                                  -> TLT 60% + SHY 37%
  ELIF (SPY 21d ret > 0) AND (JNK 21d ret < 0)       -> SPY 97% (moderate) for
       a 21-bar cooldown (locked-in flag, not re-checked weekly)
  ELSE                                                -> QQQ 97% (offensive)

The cooldown is critical: divergence is a *signal* not a *state*. We don't want
to flip back to QQQ the moment JNK ticks up; we want to ride out a full month
in defensive equity to let the credit canary recover.

Why anti-consensus
------------------
- No agent has used the divergence pattern as a SIGNAL. Existing JNK strategies
  use single-asset MA crossovers (JNK > 30d SMA, JNK > 50d SMA) — those are
  state gates. This is a differential signal between two asset classes that
  fires only when they disagree.
- gen7_commodity_equity_divergence used DBC vs SPY (commodity vs equity) — that
  failed because commodities had a secular bear in IS. JNK (HY credit) is not
  a commodity; it tracks credit conditions, which had a different cycle.
- Default-offensive (QQQ-tilted baseline) is the opposite stance from all
  existing JNK gates which default to bonds when credit weakens. By making
  the default offensive and divergence-only the de-risker, we avoid the bond
  drag that killed gen6_jnk_lqd_spy_regime variants.

What could go wrong
-------------------
- 2010-2018 had relatively few sustained SPY-up/JNK-down divergences (notable
  ones: mid-2011 European debt scare, summer-2015 China devaluation, Q4-2015
  HY energy stress, early-2018 vol spike). If the trigger fires <20 times,
  cooldowns generate few trades and the strategy is essentially QQQ-clone with
  an outer 200d gate (which is gen5_curated territory and may correlate >0.85).
- The divergence threshold may need calibration. Strict (both signs) might be
  too rare; loose (e.g. SPY > +1pct AND JNK < -0.5pct) might fire more.
- QQQ-default with rare cooldowns could hit 0.85 corr to gen7_sp500_idiosyncratic
  (which is also QQQ-correlated through SP500 mega-caps). If rejected on corr,
  the postmortem still says "credit canary as signal not state" is unique.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "QQQ", "TLT", "SHY", "JNK"]

REBALANCE_EVERY = 5            # weekly check
RET_WINDOW = 21                # 21d return for both SPY and JNK
COOLDOWN_BARS = 21             # 21 bars in defensive-equity after divergence
TREND_WINDOW = 200             # SPY 200d outer gate
EXPOSURE = 0.97
DEF_TLT = 0.60
DEF_SHY = 0.37


class CreditDivergenceCanary(Strategy):
    """Default QQQ; rotate to SPY for 21 bars when credit canary diverges from equity."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ret_window: int = RET_WINDOW,
        cooldown_bars: int = COOLDOWN_BARS,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
        def_tlt: float = DEF_TLT,
        def_shy: float = DEF_SHY,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ret_window=ret_window,
            cooldown_bars=cooldown_bars,
            trend_window=trend_window,
            exposure=exposure,
            def_tlt=def_tlt,
            def_shy=def_shy,
        )
        self.rebalance_every = int(rebalance_every)
        self.ret_window = int(ret_window)
        self.cooldown_bars = int(cooldown_bars)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self.def_tlt = float(def_tlt)
        self.def_shy = float(def_shy)
        # Internal state: bar index when last divergence was detected.
        # Strategy is in "defensive-equity (SPY)" mode while
        # ctx.idx <= self._last_divergence_idx + cooldown_bars.
        self._last_divergence_idx: int = -10**9

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.ret_window, self.trend_window) + 5
        if ctx.idx < warmup:
            return []

        # We re-check the divergence trigger DAILY (so we don't miss a
        # mid-week credit blowout) but only emit orders weekly.
        check_now = self._evaluate_signals(ctx)
        if check_now == "divergence":
            self._last_divergence_idx = ctx.idx

        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine current target based on regime and cooldown
        regime = self._current_regime(ctx)

        target: dict[str, float] = {}
        if regime == "bear":
            if "TLT" in live:
                target["TLT"] = self.def_tlt * self.exposure
            if "SHY" in live:
                target["SHY"] = self.def_shy * self.exposure
        elif regime == "defensive_eq":
            if "SPY" in live:
                target["SPY"] = self.exposure
        else:  # "offensive"
            if "QQQ" in live:
                target["QQQ"] = self.exposure
            elif "SPY" in live:
                target["SPY"] = self.exposure

        if not target:
            return []

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

    def _evaluate_signals(self, ctx: BarContext) -> str:
        """Return 'bear' | 'divergence' | 'normal' for *today*."""
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return "normal"
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window or len(spy_close) <= self.ret_window:
            return "normal"

        spy_now = float(spy_close.iloc[-1])
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        if spy_now <= spy_sma:
            return "bear"

        spy_ret = float(spy_now / spy_close.iloc[-1 - self.ret_window] - 1.0)

        try:
            jnk_hist = ctx.history("JNK")
        except KeyError:
            return "normal"
        jnk_close = jnk_hist["close"].dropna()
        if len(jnk_close) <= self.ret_window:
            return "normal"
        jnk_now = float(jnk_close.iloc[-1])
        jnk_then = float(jnk_close.iloc[-1 - self.ret_window])
        if jnk_then <= 0:
            return "normal"
        jnk_ret = jnk_now / jnk_then - 1.0

        if spy_ret > 0 and jnk_ret < 0:
            return "divergence"
        return "normal"

    def _current_regime(self, ctx: BarContext) -> str:
        sig = self._evaluate_signals(ctx)
        if sig == "bear":
            return "bear"
        # Honor cooldown: if a divergence was detected within the last
        # cooldown_bars, stay in defensive-equity even if signal is now normal.
        if (ctx.idx - self._last_divergence_idx) <= self.cooldown_bars:
            return "defensive_eq"
        return "offensive"


NAME = "opus5_credit_divergence_canary"
HYPOTHESIS = (
    "Equity-credit divergence canary: default offensive QQQ 97%; on rare days "
    "where SPY 21d>0 AND JNK 21d<0 (credit canary diverges from equity rally), "
    "rotate to SPY 97% for 21-bar cooldown; SPY<200d outer gate -> TLT 60%+SHY 37%; "
    "credit-as-signal not credit-as-state, default offensive — opposite of every "
    "JNK MA gate strategy which defaults to bonds."
)

STRATEGY = CreditDivergenceCanary()
