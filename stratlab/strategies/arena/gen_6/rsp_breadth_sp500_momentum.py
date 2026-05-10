"""RSP/SPY breadth-ratio gated SP500 momentum strategy.

Hypothesis: When RSP (equal-weight S&P500) outperforms SPY (cap-weight) on
20-day return, the broad market is participating — hold top-15 SP500 stocks
by 63d total return (equally weighted). When SPY is above 200d SMA but RSP
lags SPY, hold SPY (neutral). When SPY is below 200d SMA, hold TLT. Biweekly
rebalance.

Rationale:
  The RSP/SPY ratio is a direct measure of equal-weight vs cap-weight
  performance, which captures whether smaller/broader stocks are participating
  in the rally or if returns are being driven only by mega-caps. When
  equal-weight leads, sector breadth is wide and momentum portfolios across
  the broad SP500 tend to work better. When cap-weight leads, concentration
  risk is rising.

  Gen5 gen5_atr_momentum_etf (IS Calmar 0.74) used RSP/SPY to gate between
  ETFs (QQQ vs SPY vs TLT). This strategy extends the same signal to drive
  individual SP500 stock selection, which is a different implementation.

Diversification vs leaderboard:
  - corr to top-5 expected to be moderate (0.5-0.7) since similar IS window
    regime but different mechanism (RSP breadth vs VIX vs credit).
  - Stock selection via 63d return (not 126d, not skip-month) with RSP gate.
  - Only uses RSP breadth for risk-on differentiation within bull market.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # biweekly
MOMENTUM_WINDOW = 63     # ~3 months
RSP_SPY_WINDOW = 20      # RSP vs SPY 20d relative return
TREND_WINDOW = 200       # SPY 200d SMA
TOP_K = 15
EXPOSURE = 0.97

_RSP = "RSP"
_SPY = "SPY"


class RSPBreadthSP500Momentum(Strategy):
    """RSP/SPY 20d breadth gate + top-15 SP500 63d momentum."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        rsp_spy_window: int = RSP_SPY_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            rsp_spy_window=rsp_spy_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.rsp_spy_window = int(rsp_spy_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.momentum_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        if not bull:
            # Bear market: TLT
            closes_now = ctx.closes()
            if closes_now.empty:
                return []
            live = {s: float(p) for s, p in closes_now.items()}
            equity = ctx.portfolio_value(live)
            if equity <= 0:
                return []
            target: dict[str, float] = {}
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
            return self._build_orders(ctx, target, live, equity)

        # --- RSP vs SPY 20d breadth signal ---
        rsp_leads = False  # default to SPY mode if RSP data not available
        try:
            rsp_hist = ctx.history(_RSP)
            if rsp_hist is not None and len(rsp_hist) >= self.rsp_spy_window + 1:
                rsp_close = rsp_hist["close"].dropna()
                if len(rsp_close) >= self.rsp_spy_window:
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.rsp_spy_window] - 1.0)
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.rsp_spy_window] - 1.0)
                    if np.isfinite(rsp_ret) and np.isfinite(spy_ret):
                        rsp_leads = rsp_ret > spy_ret
        except (KeyError, IndexError):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target = {}

        if not rsp_leads:
            # Narrow breadth (cap-weight leads): hold SPY
            if _SPY in closes_now.index:
                target[_SPY] = self.exposure
        else:
            # Broad breadth (equal-weight leads): top-K momentum stocks
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                if _SPY in closes_now.index:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    # Skip ETFs and non-stocks
                    if sym in {"TLT", "GLD", "SHY", "IEF", "AGG", "JNK", "LQD",
                               "SPY", "QQQ", "IWM", "RSP", "DBC", "SSO", "TQQQ", "GDX",
                               "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY",
                               "HYG", "SMH", "SHV", "BIL"}:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < self.top_k:
                    if _SPY in closes_now.index:
                        target[_SPY] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:self.top_k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight

        return self._build_orders(ctx, target, live, equity)

    def _build_orders(
        self,
        ctx: BarContext,
        target: dict[str, float],
        live: dict[str, float],
        equity: float,
    ) -> list[Order]:
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
    return sp500_tickers() + ["TLT", "SPY", "RSP"]


NAME = "rsp_breadth_sp500_momentum"
HYPOTHESIS = (
    "RSP-SPY breadth ratio gated SP500 momentum: when RSP outperforms SPY on 20d return "
    "(broad breadth) hold top-15 SP500 by 63d return equally; when SPY above 200d but "
    "RSP underperforms hold SPY; when SPY below 200d hold TLT; biweekly rebalance; "
    "extends gen5 RSP breadth signal from ETF rotation to individual stock selection."
)

UNIVERSE = _universe

STRATEGY = RSPBreadthSP500Momentum()
