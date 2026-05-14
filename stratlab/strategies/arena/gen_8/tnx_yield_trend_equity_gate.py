"""TNX Yield Trend Equity Gate — gen_8 sonnet-10

Hypothesis: Use the 10-year Treasury yield (^TNX) trend (vs its own 200d MA)
as an equity regime signal. When TNX is BELOW its 200d MA (yields falling or
at low level → rates accommodative), hold top-15 SP500 stocks by 63d momentum.
When TNX is ABOVE its 200d MA (rising rates regime → headwind for growth),
rotate to SPY 60% + TLT 37% (blend — not fully defensive since rising rates
can still have positive equity exposure, just less concentrated).

Rationale: The level of rates relative to trend is distinct from:
- Yield curve slope (10Y-2Y spread) — already on leaderboard
- VIX level — already on leaderboard
- Credit spread (JNK/LQD) — already on leaderboard
Using the yield's own trend (vs 200d MA) identifies rate-cycle regimes.
During 2010-2018 IS window, TNX spent extended periods both below and above
its 200d MA, giving different regime exposures.

SPY 200d SMA second gate: if SPY is itself in bear, go fully to TLT.
Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOMENTUM_WINDOW = 63      # ~3 months for stock selection
YIELD_TREND_WINDOW = 200  # 200d MA for TNX yield trend
SPY_TREND_WINDOW = 200    # SPY bear gate
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_TNX = "^TNX"   # 10-year Treasury yield (signal-only)


class TNXYieldTrendEquityGate(Strategy):
    """SP500 momentum gated by 10-year yield trend (TNX vs 200d MA)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        yield_trend_window: int = YIELD_TREND_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            yield_trend_window=yield_trend_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.yield_trend_window = int(yield_trend_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.yield_trend_window, self.spy_trend_window) + 10
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

        # TNX yield trend signal
        tnx_low_rate = True  # default assumption if signal unavailable
        try:
            tnx_hist = ctx.history(_TNX)
            if tnx_hist is not None and len(tnx_hist) >= self.yield_trend_window + 2:
                tnx_close = tnx_hist["close"].dropna()
                if len(tnx_close) >= self.yield_trend_window:
                    tnx_ma = float(tnx_close.iloc[-self.yield_trend_window:].mean())
                    tnx_now = float(tnx_close.iloc[-1])
                    # Low/falling rates regime: TNX below its 200d MA
                    tnx_low_rate = tnx_now < tnx_ma
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
            # SPY in bear — fully defensive TLT
            if _TLT in live:
                target[_TLT] = self.exposure

        elif not tnx_low_rate:
            # Rising/elevated rates regime — SPY 60% + TLT 37% blend
            # (reduces concentration but maintains equity exposure)
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _TLT in live:
                target[_TLT] = self.exposure * 0.382

        else:
            # Low/falling rates regime + SPY bull — top-K SP500 momentum
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                # Fall back to SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT):
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
    return sp500_tickers() + [_TLT, _SPY, _TNX]


NAME = "tnx_yield_trend_equity_gate"
HYPOTHESIS = (
    "TNX yield level vs 200d MA equity gating: when 10-year treasury yield (TNX) is below "
    "its 200d MA (rates falling/low, risk-on) hold top-15 SP500 stocks by 63d momentum; "
    "when TNX above 200d MA (rising rates, risk-off pressure) hold SPY 60%+TLT 37%; "
    "SPY 200d bear gate to full TLT; biweekly rebalance; yield-trend as equity regime gate "
    "distinct from yield-curve-slope and VIX signals"
)

UNIVERSE = _universe

STRATEGY = TNXYieldTrendEquityGate()
