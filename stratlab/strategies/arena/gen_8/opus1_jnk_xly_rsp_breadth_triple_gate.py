"""opus-1 / gen_8 — JNK + XLY/XLP + RSP/SPY Breadth Triple-Gate

Mutation of gen8_jnk_xly_spy_triple_gate_sp500 (IS Calmar 0.75, h1/h2 0.74/1.04).

Parent uses three risk-on signals to score 0-3 and tier exposure:
  1. JNK 20d MA > 60d MA  (credit tightening)
  2. XLY 20d return > XLP 20d return  (consumer discretionary leading staples)
  3. SPY 50d MA > 150d MA  (golden-cross-ish price trend)

This variant replaces signal #3 (SPY price trend) with a *breadth* signal:
  3'. RSP/SPY ratio above its 63d MA  (equal-weight outperforming cap-weight
       → broad participation, healthy market internals)

Rationale: SPY 50/150 cross is a price-trend signal — already correlated with
many other strategies. RSP/SPY captures market breadth (Cumulatively, when
small/equal-weight outperforms large-cap, it indicates participation breadth).
Switching the third gate from price-trend to breadth gives a distinct loss-mode
profile while preserving the credit + consumer signal pair.

Same 4-tier allocation: 3/3 → top-15 inv-vol; 2/3 → top-10 equal; 1/3 →
IEF+GLD blend; 0/3 → TLT.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
STOCK_TREND_WINDOW = 50
BREADTH_MA = 63                # RSP/SPY 63d MA breadth signal
JNK_FAST_MA = 20
JNK_SLOW_MA = 60
CONSUMER_WINDOW = 20
TOP_K_HIGH = 15
TOP_K_MED = 10
EXPOSURE_HIGH = 0.97
EXPOSURE_MED = 0.85
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_GLD = "GLD"
_JNK = "JNK"
_XLY = "XLY"
_XLP = "XLP"
_RSP = "RSP"


class JnkXlyRspBreadthTripleGate(Strategy):
    """3-signal tiered SP500 momentum: JNK + XLY/XLP + RSP/SPY breadth."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        breadth_ma: int = BREADTH_MA,
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
            breadth_ma=breadth_ma,
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
        self.breadth_ma = int(breadth_ma)
        self.jnk_fast_ma = int(jnk_fast_ma)
        self.jnk_slow_ma = int(jnk_slow_ma)
        self.consumer_window = int(consumer_window)
        self.top_k_high = int(top_k_high)
        self.top_k_med = int(top_k_med)
        self.exposure_high = float(exposure_high)
        self.exposure_med = float(exposure_med)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.breadth_ma, self.jnk_slow_ma, self.momentum_window,
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

        # --- Signal 3': RSP/SPY breadth above 63d MA ---
        signal_breadth = False
        try:
            rsp_hist = ctx.history(_RSP)
            spy_hist = ctx.history(_SPY)
            if (len(rsp_hist) >= self.breadth_ma + 5 and
                    len(spy_hist) >= self.breadth_ma + 5):
                rsp_cl = rsp_hist["close"].dropna()
                spy_cl = spy_hist["close"].dropna()
                n = min(len(rsp_cl), len(spy_cl))
                if n >= self.breadth_ma + 1:
                    rsp_v = rsp_cl.values[-n:]
                    spy_v = spy_cl.values[-n:]
                    spy_safe = np.where(spy_v > 0, spy_v, np.nan)
                    ratio = rsp_v / spy_safe
                    valid = ratio[~np.isnan(ratio)]
                    if len(valid) >= self.breadth_ma + 1:
                        ratio_ma = float(np.mean(valid[-self.breadth_ma:]))
                        ratio_now = float(valid[-1])
                        signal_breadth = ratio_now > ratio_ma
        except Exception:
            pass

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

        risk_score = int(signal_jnk) + int(signal_consumer) + int(signal_breadth)
        target: dict[str, float] = {}

        if risk_score == 0:
            if _TLT in live:
                target[_TLT] = 0.97
        elif risk_score == 1:
            for sym, w in [(_IEF, 0.60), (_GLD, 0.37)]:
                if sym in live:
                    target[sym] = w * 0.97
        else:
            top_k = self.top_k_high if risk_score == 3 else self.top_k_med
            exposure = self.exposure_high if risk_score == 3 else self.exposure_med

            prices = ctx.closes_window(self.momentum_window + 10)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _TLT, _IEF, _GLD, _JNK, _XLY, _XLP, _RSP):
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
    return sp500_tickers() + [_SPY, _TLT, _IEF, _GLD, _JNK, _XLY, _XLP, _RSP]


NAME = "opus1_jnk_xly_rsp_breadth_triple_gate"
HYPOTHESIS = (
    "Mutation of jnk_xly_spy_triple_gate_sp500: replace SPY 50/150 golden cross with RSP/SPY "
    "breadth ratio above 63d MA; same JNK 20/60 + XLY/XLP signals; tiered 3-gate SP500 "
    "momentum where 3rd gate is breadth (not price-trend); biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = JnkXlyRspBreadthTripleGate()
