"""UUP Dollar Trend Gated SP500 Momentum — gen_8 sonnet-10

Hypothesis: Use the US dollar trend (UUP ETF vs its 63d SMA) as an equity
regime gate for SP500 cross-sectional momentum.

- Dollar weakening (UUP below 63d SMA): risk-on regime, foreign capital
  flows into US equities, growth momentum works → hold top-15 SP500 stocks
  by 63d return
- Dollar strengthening (UUP above 63d SMA): risk-off, international headwinds,
  growth slows → hold SPY 60% + IEF 37% (moderate defensive, not fully out)
- SPY below 200d SMA (bear): full TLT defensive

Rationale: USD strength/weakness is a macro cross-asset signal that:
1. Is NOT the same as VIX (vol-based gate)
2. Is NOT the same as credit spread (JNK/LQD, HYG gates)
3. Is NOT the same as yield curve slope (TNX-IRX, TYX)
4. Is NOT the same as TNX level vs 200d MA (just accepted)

In the IS window (2010-2018), USD had distinct trend regimes:
- Dollar weakness 2010-2011 (QE era, EM/commodity boom)
- Dollar strength 2014-2015 (Fed tapering, divergence from ECB/BOJ)
- Dollar neutral/choppy 2016-2018

These regimes have different equity momentum profiles. Dollar weakness favors
growth and international exposure; dollar strength favors quality/domestic.

The 63d MA for UUP is short enough to capture meaningful trend shifts but
long enough to avoid daily noise.

Biweekly rebalance (10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # ~3 months for stock selection
SPY_TREND_WINDOW = 200    # SPY bear gate
UUP_MA_WINDOW = 63        # 63d MA for dollar trend signal
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_UUP = "UUP"


class UUPDollarTrendSP500(Strategy):
    """SP500 momentum gated by US dollar trend (UUP vs 63d MA)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        uup_ma_window: int = UUP_MA_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            spy_trend_window=spy_trend_window,
            uup_ma_window=uup_ma_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.spy_trend_window = int(spy_trend_window)
        self.uup_ma_window = int(uup_ma_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.uup_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY bear gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # UUP dollar trend signal
        dollar_weak = True  # default: assume dollar weak (risk-on) if signal unavailable
        try:
            uup_hist = ctx.history(_UUP)
            if uup_hist is not None and len(uup_hist) >= self.uup_ma_window + 2:
                uup_close = uup_hist["close"].dropna()
                if len(uup_close) >= self.uup_ma_window + 1:
                    uup_ma = float(uup_close.iloc[-self.uup_ma_window:].mean())
                    uup_now = float(uup_close.iloc[-1])
                    # Dollar weak = UUP below its MA
                    dollar_weak = uup_now < uup_ma
        except (KeyError, Exception):
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
            # Bear: full TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        elif not dollar_weak:
            # Dollar strengthening: SPY + IEF blend (moderate defensive)
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _IEF in live:
                target[_IEF] = self.exposure * 0.382

        else:
            # Dollar weak + SPY bull: top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF, _UUP):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

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
    return sp500_tickers() + [_TLT, _SPY, _IEF, _UUP]


NAME = "uup_dollar_trend_sp500"
HYPOTHESIS = (
    "UUP dollar trend gated SP500 momentum: when UUP (dollar ETF) is below its 63d SMA "
    "(dollar weakening, risk-on for equities) hold top-15 SP500 stocks by 63d momentum; "
    "when UUP above 63d SMA (dollar strengthening, risk-off pressure on international/growth) "
    "hold SPY 60%+IEF 37%; SPY 200d bear gate to TLT; biweekly rebalance; "
    "dollar trend as equity gate distinct from all existing signals"
)

UNIVERSE = _universe

STRATEGY = UUPDollarTrendSP500()
