"""PFF/JNK credit-quality spread regime — gen_7 opus-2 (gap_finder).

Hypothesis: The relative strength between PFF (preferred stock, IG-quality
credit-sensitive) and JNK (high-yield/junk bonds) is a novel credit-quality
stress signal. When JNK outperforms PFF on 60d return, the high-yield end of
the credit curve is rallying — risk-on. When PFF leads JNK, defensive
preferreds outperforming junk indicates credit-stress regime.

Logic:
  - Compute 60d return of PFF and JNK.
  - JNK_ret > PFF_ret AND SPY > 200d SMA -> top-15 SP500 by 63d momentum.
  - PFF_ret > JNK_ret AND SPY > 200d -> SPY 50% + TLT 47% (defensive blend).
  - SPY < 200d SMA -> TLT 97% (bear regime override).

Distinction: existing strategies use JNK alone (vs MA, vs IWM, vs VIX) but
none use the credit-quality term-structure (preferreds vs high-yield).
PFF tracks investment-grade-rated preferreds while JNK tracks BB/B junk;
the spread between them is a cleaner credit-stress signal than JNK level.

Data: PFF inception 2007-03; JNK inception 2007-12. Both available across
the 2010-2018 IS window with ample history.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
RATIO_WINDOW = 60
STOCK_MOM_WINDOW = 63
TOP_K = 15
TREND_WINDOW = 200
EXPOSURE = 0.97


class PffJnkCreditQualityRegime(Strategy):
    """PFF vs JNK 60d return regime gates SP500 momentum vs SPY/TLT defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ratio_window: int = RATIO_WINDOW,
        stock_mom_window: int = STOCK_MOM_WINDOW,
        top_k: int = TOP_K,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ratio_window=ratio_window,
            stock_mom_window=stock_mom_window,
            top_k=top_k,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.ratio_window = int(ratio_window)
        self.stock_mom_window = int(stock_mom_window)
        self.top_k = int(top_k)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_mom_window, self.ratio_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d SMA bear gate
        bull_market = True
        try:
            spy_hist = ctx.history("SPY")
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    bull_market = float(spy_close.iloc[-1]) > spy_sma
        except Exception:
            pass

        # PFF vs JNK 60d return
        credit_risk_on = True  # default risk-on
        signal_ok = False
        try:
            pff_hist = ctx.history("PFF")
            jnk_hist = ctx.history("JNK")
            if (pff_hist is not None and jnk_hist is not None
                    and len(pff_hist) >= self.ratio_window
                    and len(jnk_hist) >= self.ratio_window):
                pff_close = pff_hist["close"].dropna()
                jnk_close = jnk_hist["close"].dropna()
                if (len(pff_close) >= self.ratio_window
                        and len(jnk_close) >= self.ratio_window):
                    pff_ret = float(pff_close.iloc[-1] / pff_close.iloc[-self.ratio_window] - 1.0)
                    jnk_ret = float(jnk_close.iloc[-1] / jnk_close.iloc[-self.ratio_window] - 1.0)
                    credit_risk_on = (jnk_ret > pff_ret)
                    signal_ok = True
        except Exception:
            pass

        target: dict[str, float] = {}

        if not bull_market:
            # SPY < 200d -> TLT defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif not signal_ok or not credit_risk_on:
            # Credit-stress regime -> SPY/TLT defensive blend
            for sym, w in [("SPY", 0.50), ("TLT", 0.47)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: SP500 top-K by 63d momentum
            need = self.stock_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.stock_mom_window:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.stock_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.stock_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret
                if len(scores) < self.top_k:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:self.top_k]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
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
    return sp500_tickers() + ["SPY", "TLT", "PFF", "JNK"]


NAME = "opus2_pff_jnk_credit_quality"
HYPOTHESIS = (
    "PFF/JNK credit-quality spread regime: when JNK outperforms PFF on 60d return AND SPY > 200d, "
    "hold SP500 top-15 63d momentum; when PFF leads JNK (credit stress), hold SPY 50%+TLT 47%; "
    "SPY < 200d hold TLT; biweekly rebalance; preferred-vs-HY spread is novel credit-stress signal."
)
UNIVERSE = _universe

STRATEGY = PffJnkCreditQualityRegime()
