"""gen_9 opus-1 — EFA vs US regime gate (DM-vs-US flavor).

Parent: gen9_em_us_regime_gate (IS Calmar 0.96, corr 0.665 LOW, h1=1.19 h2=0.73).
Mutation:
  - Replace EEM (emerging mkts) -> EFA (developed intl ex-US)
  - Tighten threshold: 2pp -> 1pp (more frequent risk-on firing)
  - Same regime logic:
      * EFA leads SPY by >1pp (DM risk-on, USD weak) -> top-10 SP500 by 63d
        momentum
      * SPY leads EFA (US dominance) -> SPY 60% + IEF 37%
      * both EFA and SPY negative 60d -> TLT 97%
      * SPY < 200d SMA -> TLT 97% (outer bear)

Rationale: The parent has the lowest corr-to-top5 (0.665) — preserve that
property by keeping the structural shape and only swapping the intl benchmark.
EFA covers developed markets (Europe + Japan), so the macro mechanism shifts
from "EM growth flow" to "DM monetary divergence" (when Europe/Japan outperform
US it's usually because the US dollar is weakening). The 1pp threshold makes
the gate fire more often (parent's TLT/SPY+IEF defensive bias might be over-
defensive given the calm IS regime).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_WINDOW = 63
RATIO_WINDOW = 60
INTL_LEAD_THRESHOLD = 0.01    # was 0.02
SPY_TREND_WINDOW = 200
TOP_K = 10
EXPOSURE = 0.97


class Opus1EfaUsRegimeGate(Strategy):
    """EFA vs US 60d return spread gate: SP500 momentum (DM leads) or SPY+IEF
    (US leads) or TLT (both negative); SPY 200d outer bear gate.
    """

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOM_WINDOW, RATIO_WINDOW, SPY_TREND_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND_WINDOW + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW:
            return []
        spy_sma200 = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        try:
            efa_hist = ctx.history("EFA")
        except KeyError:
            efa_hist = None

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in closes_now.index:
                target["TLT"] = EXPOSURE
        else:
            efa_ret = float("nan")
            spy_ret = float("nan")

            if efa_hist is not None and len(efa_hist) >= RATIO_WINDOW + 2:
                efa_c = efa_hist["close"].dropna()
                if len(efa_c) >= RATIO_WINDOW + 1:
                    efa_ret = float(efa_c.iloc[-1] / efa_c.iloc[-RATIO_WINDOW] - 1.0)

            if len(spy_close) >= RATIO_WINDOW + 1:
                spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-RATIO_WINDOW] - 1.0)

            if not np.isfinite(efa_ret) or not np.isfinite(spy_ret):
                if "SPY" in closes_now.index and "IEF" in closes_now.index:
                    target["SPY"] = 0.60 * EXPOSURE
                    target["IEF"] = 0.37 * EXPOSURE
            elif spy_ret < 0 and efa_ret < 0:
                if "TLT" in closes_now.index:
                    target["TLT"] = EXPOSURE
            elif efa_ret > spy_ret + INTL_LEAD_THRESHOLD:
                # DM leads US -> top-K SP500 momentum
                prices = ctx.closes_window(MOM_WINDOW + 5)
                if len(prices) < MOM_WINDOW:
                    if "SPY" in closes_now.index:
                        target["SPY"] = EXPOSURE
                else:
                    scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in ("EFA", "SPY", "IEF", "TLT"):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < MOM_WINDOW:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-MOM_WINDOW] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret

                    if len(scores) < TOP_K:
                        if "SPY" in closes_now.index:
                            target["SPY"] = EXPOSURE
                    else:
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                        longs = ranked[:TOP_K]
                        per_weight = EXPOSURE / len(longs)
                        for sym in longs:
                            target[sym] = per_weight
            else:
                # US leads DM (USD strength) -> SPY+IEF blend
                if "SPY" in closes_now.index and "IEF" in closes_now.index:
                    target["SPY"] = 0.60 * EXPOSURE
                    target["IEF"] = 0.37 * EXPOSURE

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
    return sp500_tickers() + ["TLT", "IEF", "SPY", "EFA"]


UNIVERSE = _universe

NAME = "opus1_efa_us_regime_gate"
HYPOTHESIS = (
    "Mutate gen9_em_us_regime_gate: replace EEM with EFA (developed-intl ex-US); "
    "tighten threshold to 1pp (was 2pp); when EFA leads SPY by >1pp 60d -> top-10 SP500 momentum; "
    "SPY leads EFA -> SPY60+IEF37; both negative -> TLT; SPY 200d outer bear -> TLT; "
    "DM-vs-US flow preserves low-corr property of parent (0.665)."
)

STRATEGY = Opus1EfaUsRegimeGate()
