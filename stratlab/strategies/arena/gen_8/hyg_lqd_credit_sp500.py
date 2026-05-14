"""HYG/LQD Credit Spread SP500 Momentum — gen_8 sonnet-8

Hypothesis: When HY bonds (HYG) are outperforming IG bonds (LQD) on a 20d
return basis (credit spreads tightening = risk-on) AND SPY > 200d SMA, hold
top-15 SP500 stocks by 63d momentum; equal-weight. Hold IEF when either
gate fails. Biweekly rebalance.

Rationale: The HYG vs LQD 20d return DIFFERENTIAL is a forward-looking
credit signal — when investors are rotating from investment-grade to
high-yield bonds, credit spreads are tightening, which leads equity markets.
This is distinct from:
- JNK level vs 30d MA (absolute level, not spread differential)
- HYG/LQD ratio (this uses the RETURN differential, not price ratio level)

The signal is: HYG_20d_return > LQD_20d_return (i.e. HY outperforming IG)
signals appetite for risk, leading to equity outperformance. When IG leads,
institutions are de-risking and equities should be avoided.

Distinct from leaderboard:
- gen6_hy_credit_qqq_rotation: JNK >30d SMA (level signal) → QQQ (not stocks)
- gen6_rp_credit_tilt: JNK signal on risk-parity basket
- gen7_opus2_pff_jnk_credit_quality: PFF vs JNK (preferred vs HY)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
REBALANCE_EVERY = 10          # biweekly
CREDIT_WINDOW = 20            # HYG vs LQD return comparison window
MOM_WINDOW = 63               # stock momentum window
TREND_WINDOW = 200            # SPY bear gate
TOP_K = 15
EXPOSURE = 0.97
_HYG = "HYG"
_LQD = "LQD"
_SPY = "SPY"
_IEF = "IEF"


class HYGLQDCreditSP500(Strategy):
    """HYG-over-LQD credit tightening signal gating SP500 stock momentum."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        credit_window: int = CREDIT_WINDOW,
        mom_window: int = MOM_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            credit_window=credit_window,
            mom_window=mom_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.credit_window = int(credit_window)
        self.mom_window = int(mom_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.mom_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # ---- SPY trend gate ----
        bull = False
        try:
            spy_hist = ctx.history(_SPY)
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_now = float(spy_close.iloc[-1])
                bull = spy_now > spy_sma
        except (KeyError, Exception):
            bull = False

        # ---- Credit spread gate: HYG 20d return > LQD 20d return ----
        credit_ok = False
        try:
            hyg_hist = ctx.history(_HYG)
            lqd_hist = ctx.history(_LQD)
            hyg_close = hyg_hist["close"].dropna()
            lqd_close = lqd_hist["close"].dropna()
            if len(hyg_close) >= self.credit_window + 1 and len(lqd_close) >= self.credit_window + 1:
                hyg_ret = float(hyg_close.iloc[-1] / hyg_close.iloc[-self.credit_window] - 1.0)
                lqd_ret = float(lqd_close.iloc[-1] / lqd_close.iloc[-self.credit_window] - 1.0)
                if np.isfinite(hyg_ret) and np.isfinite(lqd_ret):
                    credit_ok = hyg_ret > lqd_ret
        except (KeyError, Exception):
            credit_ok = False

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if bull and credit_ok:
            # Risk-on: top-K SP500 stocks by 63d momentum
            prices = ctx.closes_window(self.mom_window + 5)
            if len(prices) < self.mom_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                if sym in (_SPY, _IEF, _HYG, _LQD):
                    continue
                if sym not in live:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.mom_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < 5:
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    if sym in live:
                        target[sym] = per_weight
        else:
            # Defensive: IEF
            if _IEF in live:
                target[_IEF] = self.exposure

        # ---- Build orders ----
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_IEF, _SPY, _HYG, _LQD]


NAME = "hyg_lqd_credit_sp500"
HYPOTHESIS = (
    "HYG/LQD credit spread ratio gated SP500 momentum: when HYG 20d return > LQD 20d return "
    "(credit tightening, risk-on) AND SPY > 200d SMA, hold top-15 SP500 stocks by 63d momentum; "
    "equal-weight; IEF defensive; biweekly rebalance; uses HY-vs-IG spread direction as "
    "credit signal not JNK level"
)

UNIVERSE = _universe

STRATEGY = HYGLQDCreditSP500()
