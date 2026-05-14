"""JNK Credit + XLY Consumer Confidence Dual-Gate SP500 Momentum — gen_8 sonnet-5

Hypothesis: Combine two orthogonal risk signals as dual gates for SP500 stock
momentum:

Gate 1 (Credit health): JNK 20d MA vs 60d MA crossover.
  - JNK above 60d MA = credit spreads tightening = risk-on.
  - JNK below 60d MA = spreads widening = credit stress.

Gate 2 (Consumer confidence): XLY 20d return > XLP 20d return.
  - When consumer discretionary outperforms consumer staples, households are
    spending on wants vs needs = confidence indicator.
  - When staples lead discretionary, defensive consumption = risk-off.

Regime logic:
  - Both gates risk-on: hold top-15 SP500 stocks by 63d momentum above
    individual 100d SMA; equal-weight; 97% exposure. (Full risk-on)
  - Credit ok but consumer cautious (XLP leads): hold top-10 SP500 by
    63d momentum but reduce to 80% exposure. (Moderate risk)
  - Credit weak (JNK below MA): hold IEF 60% + GLD 37%. (Defensive)
  - SPY below 200d SMA (outer bear): hold TLT 97%.

Rationale:
  - XLY/XLP consumer relative strength has NOT been successfully combined with
    SP500 stock selection on the leaderboard. Prior gen_7 attempt (xly_xlp_consumer_regime)
    used it for pure ETF rotation (QQQ/SPY/TLT) and got IS Calmar 0.412 —
    the signal alone wasn't strong enough. Combined with JNK credit (high
    information signal per gen_6/7 history) as a second gate, the conjunction
    should filter out false positives more cleanly.
  - JNK 20d/60d MA crossover (not JNK level vs 30d SMA which is already on
    leaderboard) captures credit TREND, not just JNK-above-level.
  - The moderate-risk tier (credit ok, consumer cautious) reduces exposure
    to 80% rather than fully defensive — prevents over-sitting-out in 2010-18 bull.

IS window 2010-2018 coverage:
  - Credit risk-on: mostly true 2010-2014 (post-crisis recovery), partial 2016-2018.
  - Consumer risk-on: mostly true 2010-2018 (strong US consumer, housing recovery).
  - The dual conjunction will be risk-on ~60-70% of IS window days.

Rebalance: every 10 bars (biweekly) for sufficient trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10          # biweekly
MOMENTUM_WINDOW = 63          # ~3 months
STOCK_TREND_WINDOW = 100      # per-stock 100d SMA
TREND_WINDOW = 200            # SPY 200d SMA outer gate
JNK_FAST_MA = 20              # JNK 20d MA
JNK_SLOW_MA = 60              # JNK 60d MA
CONSUMER_WINDOW = 20          # XLY vs XLP 20d return comparison
TOP_K_FULL = 15               # full risk-on: top-15
TOP_K_MOD = 10                # moderate: top-10
EXPOSURE_FULL = 0.97
EXPOSURE_MOD = 0.80
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_GLD = "GLD"
_JNK = "JNK"
_XLY = "XLY"
_XLP = "XLP"


class JnkXlyDualGateSP500Momentum(Strategy):
    """JNK credit + XLY/XLP consumer dual-gate SP500 momentum strategy."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        trend_window: int = TREND_WINDOW,
        jnk_fast_ma: int = JNK_FAST_MA,
        jnk_slow_ma: int = JNK_SLOW_MA,
        consumer_window: int = CONSUMER_WINDOW,
        top_k_full: int = TOP_K_FULL,
        top_k_mod: int = TOP_K_MOD,
        exposure_full: float = EXPOSURE_FULL,
        exposure_mod: float = EXPOSURE_MOD,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stock_trend_window=stock_trend_window,
            trend_window=trend_window,
            jnk_fast_ma=jnk_fast_ma,
            jnk_slow_ma=jnk_slow_ma,
            consumer_window=consumer_window,
            top_k_full=top_k_full,
            top_k_mod=top_k_mod,
            exposure_full=exposure_full,
            exposure_mod=exposure_mod,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stock_trend_window = int(stock_trend_window)
        self.trend_window = int(trend_window)
        self.jnk_fast_ma = int(jnk_fast_ma)
        self.jnk_slow_ma = int(jnk_slow_ma)
        self.consumer_window = int(consumer_window)
        self.top_k_full = int(top_k_full)
        self.top_k_mod = int(top_k_mod)
        self.exposure_full = float(exposure_full)
        self.exposure_mod = float(exposure_mod)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.jnk_slow_ma, self.momentum_window,
                     self.stock_trend_window) + 10
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

        # --- SPY 200d SMA outer bear gate ---
        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_cl = spy_hist["close"].dropna()
        if len(spy_cl) < self.trend_window:
            return []
        spy_bull = float(spy_cl.iloc[-1]) > float(spy_cl.iloc[-self.trend_window:].mean())

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: TLT 97%
            if _TLT in live:
                target[_TLT] = 0.97
        else:
            # --- Gate 1: JNK credit (20d MA vs 60d MA) ---
            credit_ok = False
            try:
                jnk_hist = ctx.history(_JNK)
                if len(jnk_hist) >= self.jnk_slow_ma + 5:
                    jnk_cl = jnk_hist["close"].dropna()
                    if len(jnk_cl) >= self.jnk_slow_ma:
                        jnk_fast = float(jnk_cl.iloc[-self.jnk_fast_ma:].mean())
                        jnk_slow = float(jnk_cl.iloc[-self.jnk_slow_ma:].mean())
                        credit_ok = jnk_fast > jnk_slow
            except Exception:
                pass

            # --- Gate 2: XLY vs XLP consumer confidence (20d return) ---
            consumer_ok = False
            try:
                xly_hist = ctx.history(_XLY)
                xlp_hist = ctx.history(_XLP)
                if (len(xly_hist) >= self.consumer_window + 5 and
                        len(xlp_hist) >= self.consumer_window + 5):
                    xly_cl = xly_hist["close"].dropna()
                    xlp_cl = xlp_hist["close"].dropna()
                    if (len(xly_cl) >= self.consumer_window + 1 and
                            len(xlp_cl) >= self.consumer_window + 1):
                        xly_ret = float(xly_cl.iloc[-1] / xly_cl.iloc[-self.consumer_window - 1] - 1.0)
                        xlp_ret = float(xlp_cl.iloc[-1] / xlp_cl.iloc[-self.consumer_window - 1] - 1.0)
                        consumer_ok = xly_ret > xlp_ret
            except Exception:
                pass

            if not credit_ok:
                # Credit weak: IEF 60% + GLD 37%
                for sym, w in [(_IEF, 0.60), (_GLD, 0.37)]:
                    if sym in live:
                        target[sym] = w * 0.97
            else:
                # Credit ok — compute SP500 momentum
                prices = ctx.closes_window(self.momentum_window + 10)
                if len(prices) < self.momentum_window:
                    return []

                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _GLD, _JNK, _XLY, _XLP):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _IEF in live:
                        target[_IEF] = 0.97
                else:
                    if consumer_ok:
                        # Full risk-on: credit ok + consumer confident = top-15 stocks
                        top_k = self.top_k_full
                        exposure = self.exposure_full
                    else:
                        # Moderate: credit ok but consumer cautious = top-10 stocks at 80%
                        top_k = self.top_k_mod
                        exposure = self.exposure_mod

                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                    # Per-stock 100d SMA trend filter
                    selected: list[str] = []
                    for sym in ranked:
                        if len(selected) >= top_k:
                            break
                        sh = ctx.history(sym)
                        if len(sh) < self.stock_trend_window:
                            continue
                        sc = sh["close"].dropna()
                        if len(sc) < self.stock_trend_window:
                            continue
                        sma = float(sc.iloc[-self.stock_trend_window:].mean())
                        price = live.get(sym, 0.0)
                        if price > sma:
                            selected.append(sym)

                    if not selected:
                        if _IEF in live:
                            target[_IEF] = 0.97
                    else:
                        per_w = exposure / len(selected)
                        for sym in selected:
                            if sym in live:
                                target[sym] = per_w

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_SPY, _TLT, _IEF, _GLD, _JNK, _XLY, _XLP]


NAME = "jnk_xly_dual_gate_sp500_momentum"
HYPOTHESIS = (
    "JNK 20d/60d MA credit gate + XLY vs XLP 20d return consumer-confidence gate for SP500 "
    "momentum: both risk-on -> top-15 SP500 63d momentum above 100d SMA equal-weight 97%; "
    "credit ok but consumer cautious -> top-10 SP500 at 80% exposure; "
    "credit weak -> IEF 60%+GLD 37%; SPY 200d bear gate -> TLT; biweekly rebalance; "
    "dual orthogonal gate distinct from single JNK-level or VIX-level gates on leaderboard"
)

UNIVERSE = _universe

STRATEGY = JnkXlyDualGateSP500Momentum()
