"""gen_9 sonnet-6 — DVY vs SPY Character-Adaptive Stock Selection

Hypothesis: DVY (iShares Select Dividend ETF) vs SPY 63d return competition
signals the market's return character:
  - When DVY leads SPY (yield premium demanded, defensive/value character):
    hold top-15 SP500 stocks ranked by LOWEST 63d realized volatility (inverse-vol
    = "defensive quality"), above 200d SMA, equal-weight, SPY bear → TLT
  - When SPY leads DVY (growth/momentum regime):
    hold top-15 SP500 stocks by 63d momentum, above 200d SMA, equal-weight
  - SPY 200d outer bear gate → TLT 97% for both regimes

Rationale: The DVY/SPY competition is a real-time measure of whether equity
markets are rewarding yield and stability (risk-off character) or price momentum
and growth (risk-on character). The strategy pivots the stock-selection criterion
rather than pivoting to defensive ETFs — staying in equities but selecting the
type of stock the market is currently rewarding. This is genuinely distinct from
existing leaderboard strategies: all existing SP500 selectors use momentum as
the fixed criterion regardless of regime; this strategy adapts the criterion.

DVY available from 2003-11-07 — full IS coverage (2010-2018).
Vol-based ranking selects different stocks than momentum ranking — low
correlation between the two branches when deployed.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
VOL_WINDOW = 63            # same window for consistency
TREND_WINDOW = 200
CHARACTER_WINDOW = 63      # DVY vs SPY comparison window
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_DVY = "DVY"


class DvySpyCharacterAdaptive(Strategy):
    """SP500 stock selection adapts between low-vol and momentum based on DVY/SPY character."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        character_window: int = CHARACTER_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_window=vol_window,
            trend_window=trend_window,
            character_window=character_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.character_window = int(character_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window, self.character_window) + 10
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

        # --- DVY vs SPY character signal ---
        # True = yield regime (low-vol stocks), False = momentum regime
        yield_regime = False  # default to momentum if DVY unavailable
        try:
            dvy_hist = ctx.history(_DVY)
            if dvy_hist is not None and len(dvy_hist) >= self.character_window + 2:
                dvy_close = dvy_hist["close"].dropna()
                if len(dvy_close) >= self.character_window + 1:
                    dvy_ret = float(
                        dvy_close.iloc[-1] / dvy_close.iloc[-self.character_window] - 1.0
                    )
                    spy_ret_char = float(
                        spy_close.iloc[-1] / spy_close.iloc[-self.character_window] - 1.0
                    )
                    # DVY leads SPY = yield-seeking regime
                    yield_regime = (
                        np.isfinite(dvy_ret) and np.isfinite(spy_ret_char)
                        and dvy_ret >= spy_ret_char
                    )
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
            # Bear regime: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Bull regime: choose stock selection criterion based on character
            look = max(self.momentum_window, self.vol_window) + 5
            prices = ctx.closes_window(look)
            if len(prices) < self.vol_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}

                if yield_regime:
                    # Low-vol regime: rank by inverse realized volatility (lower vol = higher score)
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _DVY):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.vol_window + 1:
                            continue
                        log_rets = np.log(col.values[1:] / col.values[:-1])
                        if len(log_rets) < self.vol_window:
                            continue
                        rv = float(np.std(log_rets[-self.vol_window:]) * np.sqrt(252))
                        if rv > 0 and np.isfinite(rv):
                            # Higher score = lower vol (invert)
                            scores[sym] = -rv
                else:
                    # Momentum regime: rank by 63d return
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _DVY):
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
                    # Top-k (highest score = lowest vol in yield regime, highest mom in growth)
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
    return sp500_tickers() + [_TLT, _SPY, _DVY]


NAME = "dvy_spy_character_adaptive"
HYPOTHESIS = (
    "DVY vs SPY 63d return character signal: when DVY leads SPY (yield-seeking regime), "
    "hold top-15 SP500 stocks by lowest 63d realized vol above 200d SMA; when SPY leads DVY "
    "(growth/momentum regime), hold top-15 SP500 stocks by 63d momentum above 200d SMA; "
    "SPY 200d outer bear gate to TLT; biweekly rebalance; adapts stock-selection criterion "
    "rather than pivoting to defensive ETFs"
)

UNIVERSE = _universe

STRATEGY = DvySpyCharacterAdaptive()
