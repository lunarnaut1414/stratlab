"""JNK + XLY/XLP + SPY Triple-Gate Tiered SP500 Momentum — gen_8 sonnet-5

Hypothesis: Build a composite 3-signal risk score and tier SP500 momentum
exposure accordingly.

Signal 1 (Credit): JNK 20d MA > JNK 60d MA (credit spreads tightening)
Signal 2 (Consumer): XLY 20d return > XLP 20d return (discretionary leads staples)
Signal 3 (Trend): SPY 50d MA > SPY 150d MA (medium-term golden cross)

Risk tiers:
  - 3/3 signals risk-on: top-15 SP500 by 63d momentum above individual 50d SMA,
    inverse-vol weighted, 97% exposure. (Maximum conviction)
  - 2/3 signals risk-on: top-10 SP500 by 63d momentum above individual 50d SMA,
    equal-weight, 85% exposure. (Moderate conviction)
  - 1/3 signals risk-on: IEF 60% + GLD 37% at 97%. (Defensive)
  - 0/3 signals risk-on: TLT 97%. (Maximum defensive)

Rationale:
  - Extension of the accepted gen8_jnk_xly_dual_gate_sp500_momentum (corr 0.81).
    Same 2 signals PLUS a 3rd (SPY 50d/150d MA golden cross = faster than 200d).
    The 3rd signal differentiates the regime from pure defensive to tiered.
  - Using SPY 50d/150d instead of 200d: the faster MA gives earlier warning on
    trend breaks (Q3 2011, Aug 2015) and earlier re-entry (Q1 2012, Q4 2015).
  - 4-tier allocation instead of 3-tier: the 1/3-signals-risk-on bucket routes
    to IEF+GLD (not just reducing equity) — avoids holding underperforming stocks.
  - IS window 2010-2018: 3/3 risk-on majority of time (post-crisis expansion
    + strong consumer + equities bull). 2/3 regime during brief JNK or consumer
    wobbles. 0/3 only in severe episodes (Aug 2011, Dec 2018).

Correlation risk: corr with dual-gate variant expected ~0.8-0.85 (similar
equity selection engine) but the different tiering creates enough differentiation
in low-risk-on regimes.

Rebalance: every 10 bars (biweekly) for 2000+ trades over IS window.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
STOCK_TREND_WINDOW = 50       # per-stock 50d SMA
SPY_FAST_MA = 50              # SPY 50d MA
SPY_SLOW_MA = 150             # SPY 150d MA (golden cross)
JNK_FAST_MA = 20
JNK_SLOW_MA = 60
CONSUMER_WINDOW = 20
TOP_K_HIGH = 15               # 3/3 signal
TOP_K_MED = 10                # 2/3 signal
EXPOSURE_HIGH = 0.97
EXPOSURE_MED = 0.85
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_GLD = "GLD"
_JNK = "JNK"
_XLY = "XLY"
_XLP = "XLP"


class JnkXlySpyTripleGateSP500(Strategy):
    """3-signal tiered SP500 momentum: JNK + XLY/XLP + SPY 50/150 MA cross."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        spy_fast_ma: int = SPY_FAST_MA,
        spy_slow_ma: int = SPY_SLOW_MA,
        jnk_fast_ma: int = JNK_FAST_MA,
        jnk_slow_ma: int = JNK_SLOW_MA,
        consumer_window: int = CONSUMER_WINDOW,
        top_k_high: int = TOP_K_HIGH,
        top_k_med: int = TOP_K_MED,
        exposure_high: float = EXPOSURE_HIGH,
        exposure_med: float = EXPOSURE_MED,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stock_trend_window=stock_trend_window,
            spy_fast_ma=spy_fast_ma,
            spy_slow_ma=spy_slow_ma,
            jnk_fast_ma=jnk_fast_ma,
            jnk_slow_ma=jnk_slow_ma,
            consumer_window=consumer_window,
            top_k_high=top_k_high,
            top_k_med=top_k_med,
            exposure_high=exposure_high,
            exposure_med=exposure_med,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stock_trend_window = int(stock_trend_window)
        self.spy_fast_ma = int(spy_fast_ma)
        self.spy_slow_ma = int(spy_slow_ma)
        self.jnk_fast_ma = int(jnk_fast_ma)
        self.jnk_slow_ma = int(jnk_slow_ma)
        self.consumer_window = int(consumer_window)
        self.top_k_high = int(top_k_high)
        self.top_k_med = int(top_k_med)
        self.exposure_high = float(exposure_high)
        self.exposure_med = float(exposure_med)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_slow_ma, self.jnk_slow_ma, self.momentum_window,
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

        # --- Signal 3: SPY 50d vs 150d MA ---
        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.spy_slow_ma + 5:
            return []
        spy_cl = spy_hist["close"].dropna()
        if len(spy_cl) < self.spy_slow_ma:
            return []
        spy_fast_val = float(spy_cl.iloc[-self.spy_fast_ma:].mean())
        spy_slow_val = float(spy_cl.iloc[-self.spy_slow_ma:].mean())
        signal_spy = spy_fast_val > spy_slow_val

        # --- Signal 1: JNK 20d vs 60d MA ---
        signal_jnk = False
        try:
            jnk_hist = ctx.history(_JNK)
            if len(jnk_hist) >= self.jnk_slow_ma + 5:
                jnk_cl = jnk_hist["close"].dropna()
                if len(jnk_cl) >= self.jnk_slow_ma:
                    signal_jnk = float(jnk_cl.iloc[-self.jnk_fast_ma:].mean()) > \
                                  float(jnk_cl.iloc[-self.jnk_slow_ma:].mean())
        except Exception:
            pass

        # --- Signal 2: XLY vs XLP 20d return ---
        signal_consumer = False
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
                    signal_consumer = xly_ret > xlp_ret
        except Exception:
            pass

        risk_score = int(signal_jnk) + int(signal_consumer) + int(signal_spy)
        target: dict[str, float] = {}

        if risk_score == 0:
            # All signals risk-off: TLT 97%
            if _TLT in live:
                target[_TLT] = 0.97
        elif risk_score == 1:
            # Only 1 signal risk-on: IEF 60% + GLD 37%
            for sym, w in [(_IEF, 0.60), (_GLD, 0.37)]:
                if sym in live:
                    target[sym] = w * 0.97
        else:
            # 2 or 3 signals risk-on: hold SP500 momentum stocks
            top_k = self.top_k_high if risk_score == 3 else self.top_k_med
            exposure = self.exposure_high if risk_score == 3 else self.exposure_med

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
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                if risk_score == 3:
                    # High conviction: per-stock 50d SMA + inverse-vol weight
                    sel_vols: dict[str, float] = {}
                    for sym in ranked:
                        if len(sel_vols) >= top_k:
                            break
                        sh = ctx.history(sym)
                        if len(sh) < self.stock_trend_window + 5:
                            continue
                        sc = sh["close"].dropna()
                        if len(sc) < self.stock_trend_window:
                            continue
                        sma = float(sc.iloc[-self.stock_trend_window:].mean())
                        price = live.get(sym, 0.0)
                        if price <= sma:
                            continue
                        # compute realized vol
                        if len(sc) < 22:
                            iv = 1.0
                        else:
                            rets = sc.pct_change().dropna().iloc[-21:]
                            vol = float(rets.std())
                            iv = 1.0 / vol if vol > 1e-8 else 1.0
                        sel_vols[sym] = iv

                    if not sel_vols:
                        if _IEF in live:
                            target[_IEF] = 0.97
                    else:
                        total_iv = sum(sel_vols.values())
                        for sym, iv in sel_vols.items():
                            if sym in live:
                                target[sym] = (iv / total_iv) * exposure
                else:
                    # Moderate conviction: per-stock 50d SMA + equal-weight
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


NAME = "jnk_xly_spy_triple_gate_sp500"
HYPOTHESIS = (
    "SP500 momentum with composite 3-factor risk score (JNK 20d/60d MA + XLY vs XLP 20d return "
    "+ SPY 50d/150d MA cross): 3/3 risk-on -> top-15 SP500 63d momentum above 50d SMA "
    "inverse-vol 97%; 2/3 risk-on -> top-10 at 85%; 1/3 risk-on -> IEF 60%+GLD 37%; "
    "0/3 risk-on -> TLT; biweekly rebalance; tiered 3-gate SP500 momentum distinct from dual-gate"
)

UNIVERSE = _universe

STRATEGY = JnkXlySpyTripleGateSP500()
