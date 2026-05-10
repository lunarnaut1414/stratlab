"""Bond-equity ratio regime timing strategy.

Hypothesis:
  Use the TLT/SPY price ratio as a risk-on/risk-off signal:
  - When the TLT/SPY ratio is below its 50-day moving average (risk-on),
    hold the top-10 SP500 momentum stocks (63-day return), equally weighted.
  - When the TLT/SPY ratio is above its 50-day moving average (risk-off),
    hold TLT 60% + GLD 40% as a defensive allocation.
  - Rebalance every 10 trading days.

Rationale: The TLT/SPY ratio captures risk-appetite shifts. When bonds
outperform equities relative to recent history, it signals elevated risk
aversion; when equities outperform, it signals risk-on appetite. Momentum
within SP500 is then used to select which specific stocks to hold in
risk-on mode. GLD is added in defensive mode as it often benefits from
flight-to-safety dynamics alongside bonds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10  # bars
MOMENTUM_WINDOW = 63  # ~3 months
RATIO_MA_WINDOW = 50  # 50-day MA on TLT/SPY ratio
TOP_K = 10
EXPOSURE = 0.97


class BondEquityRegime(Strategy):
    """Risk-on/risk-off rotation gated by TLT/SPY ratio vs 50d MA."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        ratio_ma_window: int = RATIO_MA_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            ratio_ma_window=ratio_ma_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = rebalance_every
        self.momentum_window = momentum_window
        self.ratio_ma_window = ratio_ma_window
        self.top_k = top_k
        self.exposure = exposure

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.ratio_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # Get TLT and SPY history to compute TLT/SPY ratio
        tlt_hist = ctx.history("TLT")
        spy_hist = ctx.history("SPY")

        if len(tlt_hist) < self.ratio_ma_window + 5 or len(spy_hist) < self.ratio_ma_window + 5:
            return []

        tlt_close = tlt_hist["close"].iloc[-(self.ratio_ma_window + 5):]
        spy_close = spy_hist["close"].iloc[-(self.ratio_ma_window + 5):]

        # Align by index
        ratio = (tlt_close / spy_close).dropna()
        if len(ratio) < self.ratio_ma_window:
            return []

        ratio_ma = float(ratio.rolling(self.ratio_ma_window).mean().iloc[-1])
        current_ratio = float(ratio.iloc[-1])

        if not np.isfinite(ratio_ma) or not np.isfinite(current_ratio):
            return []

        # risk_off = bonds outperforming vs recent trend
        risk_off = current_ratio > ratio_ma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live_closes_dict = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes_dict)
        if portfolio_value <= 0:
            return []

        target: dict[str, float] = {}

        if risk_off:
            # Defensive: TLT 60%, GLD 40%
            for sym, weight in [("TLT", 0.6), ("GLD", 0.4)]:
                if sym in closes_now.index:
                    target[sym] = weight * self.exposure
        else:
            # Risk-on: top-K momentum stocks from sp500
            prices = ctx.closes_window(self.momentum_window + 5)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if len(scores) < self.top_k:
                return []

            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            longs = ranked[:self.top_k]
            per_weight = self.exposure / len(longs)
            for sym in longs:
                target[sym] = per_weight

        # Build orders
        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
        for sym, weight in target.items():
            price = live_closes_dict.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "GLD", "SPY"]


NAME = "bond_equity_regime"
HYPOTHESIS = (
    "Bond-equity ratio regime timing: use TLT/SPY ratio 50d MA as risk-on/risk-off switch; "
    "in risk-on hold top-10 SP500 momentum stocks (63d return); "
    "in risk-off hold TLT 60% + GLD 40%. Rebalance every 10 bars."
)
UNIVERSE = _universe

STRATEGY = BondEquityRegime()
