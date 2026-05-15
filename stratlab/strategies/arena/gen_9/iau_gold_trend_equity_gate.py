"""gen_9 sonnet-6 — IAU Gold Trend as Equity Regime Gate

Hypothesis: Gold's price trend (IAU vs its 63d MA) as a macro regime signal:
  - When IAU BELOW 63d MA (gold weakening = risk appetite, equities bid):
    hold top-15 SP500 stocks by 126d-skip-21d momentum, equal-weight.
  - When IAU ABOVE 63d MA (gold rising = risk-off, inflation, or safe-haven
    demand): rotate to TLT 60% + LQD 37% (duration + credit quality blend).
  - SPY 200d SMA outer bear gate → TLT 97%.

Rationale: Gold is a classic risk-off safe-haven. When gold trends higher,
institutional demand for hard assets signals macro concern (inflation, political
risk, dollar weakness, or equity market stress). Conversely, gold weakness
during SPY bull regimes is a signal of confidence in growth assets. This
macro-behavioral signal is distinct from:
  - VIX level (measures implied vol, not gold demand)
  - Credit spreads (JNK/LQD — corporate credit vs gold sentiment)
  - Yield curve (rate expectations vs commodity/inflation expectations)

Gold defensive → TLT+LQD provides duration + credit quality diversification
(different from TLT-only defensive used in most strategies). LQD adds
investment-grade corporate bond exposure which performs well in slow-growth/
low-inflation periods when gold is also in demand.

IAU has full IS coverage (inception 2005-01-28).
Skip-month momentum (126d-21d) selects different stocks from pure 63d momentum.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10   # biweekly
MOM_LONG = 126
MOM_SKIP = 21
GOLD_MA_WINDOW = 63
TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_LQD = "LQD"
_IAU = "IAU"


class IauGoldTrendEquityGate(Strategy):
    """SP500 skip-month momentum gated by gold trend (IAU vs 63d MA)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        gold_ma_window: int = GOLD_MA_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            gold_ma_window=gold_ma_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.gold_ma_window = int(gold_ma_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.mom_long, self.gold_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- IAU gold trend gate ---
        gold_weak = True  # default risk-on if unavailable
        try:
            iau_hist = ctx.history(_IAU)
            if iau_hist is not None and len(iau_hist) >= self.gold_ma_window + 2:
                iau_close = iau_hist["close"].dropna()
                if len(iau_close) >= self.gold_ma_window + 1:
                    gold_ma = float(iau_close.iloc[-self.gold_ma_window:].mean())
                    gold_now = float(iau_close.iloc[-1])
                    # Gold below MA = gold weakening = risk-on
                    gold_weak = gold_now < gold_ma
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not gold_weak:
            # Gold rising (risk-off): TLT + LQD defensive blend
            if _TLT in live:
                target[_TLT] = self.exposure * 0.618
            if _LQD in live:
                target[_LQD] = self.exposure * 0.379
        else:
            # Gold weak (risk-on): skip-month SP500 momentum
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _LQD, _IAU):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue
                    p_start = float(col.iloc[-self.mom_long])
                    p_end = float(col.iloc[-self.mom_skip])
                    if p_start <= 0:
                        continue
                    skip_mom = p_end / p_start - 1.0
                    if np.isfinite(skip_mom):
                        scores[sym] = skip_mom

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        if sym in live:
                            target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY, _LQD, _IAU]


NAME = "iau_gold_trend_equity_gate"
HYPOTHESIS = (
    "IAU gold trend as equity regime gate: when IAU below 63d MA (gold weakening = risk "
    "appetite, equities bid), hold top-15 SP500 stocks by 126d-skip-21d momentum; when IAU "
    "above 63d MA (gold rising = risk-off / inflation), hold TLT 60%+LQD 37%; SPY 200d outer "
    "bear gate to TLT; biweekly rebalance; gold trend as regime signal distinct from "
    "VIX/credit/yield-curve signals"
)

UNIVERSE = _universe

STRATEGY = IauGoldTrendEquityGate()
