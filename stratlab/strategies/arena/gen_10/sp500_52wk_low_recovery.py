"""SP500 52-week-low recovery momentum.

Hypothesis:
    Most momentum strategies pursue winners — stocks near their 52-week HIGH.
    This strategy inverts the approach: find stocks near their 52-week LOW but
    with improving short-term price action (above their 63d SMA), capturing
    early mean-reversion in beaten-down quality companies.

    Rationale:
    - Stocks within 20% of their 52-week low have been severely punished
    - If they are simultaneously ABOVE their 63d SMA, the damage is healing
    - Ranked by their 21d return (short-term recovery strength), these are
      stocks at the INFLECTION POINT between breakdown and recovery
    - This is structurally orthogonal to all existing momentum strategies which
      buy recent strength; this buys recent recovery from recent weakness

    Design:
    - Screen: price within 120% of 52-week low (i.e., within 20% of the low)
    - Quality gate: stock must be above its own 63d SMA (structure improving)
    - Rank: by 21d total return (short-term recovery momentum)
    - Hold top-10 equal-weight (broader diversification given higher vol)
    - SPY 200d SMA outer gate: IEF defensive in bear markets
    - Biweekly rebalance (10 bars)

Differentiation:
    - gen6_nearhi_momentum_quality: buys stocks within 5% of 52-week HIGH — opposite
    - All gen5-9 strategies: rank by 63d or 126d momentum (recent winners)
    - This ranks by 21d recovery return within the 52wk-low cohort (beaten losers)
    - The 52wk-low + above-63d-SMA combination is analogous to a "distressed quality"
      screen: stock has been hit hard but structure is turning
    - Expected to have low correlation with winner-momentum strategies because the
      stock universe selected (near-52wk-low) is the exact opposite pool

Notes on risk:
    - Higher risk per name (near-52wk-low stocks have higher vol) → equal-weight
      with top-10 provides diversification
    - SPY 200d gate prevents buying beaten-down stocks in a falling market
    - 63d SMA requirement prevents buying truly broken stocks in free-fall
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
RECOVERY_WINDOW = 21       # 21d short-term recovery ranking
HIGH_LOW_WINDOW = 252      # 52-week high/low lookback
SMA_WINDOW_STOCK = 63      # per-stock SMA for quality gate
SPY_TREND_WINDOW = 200     # outer bear gate
TOP_K = 10                 # equal-weight top-10 recovery candidates
EXPOSURE = 0.97
LOW_PROXIMITY = 1.20       # within 20% of 52wk low (price / low <= 1.20)


class SP500FiftyTwoWeekLowRecovery(Strategy):
    """SP500 21d recovery momentum among stocks near 52-week low and above 63d SMA.

    Selects beaten-down stocks at inflection point; SPY 200d gate; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        recovery_window: int = RECOVERY_WINDOW,
        high_low_window: int = HIGH_LOW_WINDOW,
        sma_window_stock: int = SMA_WINDOW_STOCK,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        low_proximity: float = LOW_PROXIMITY,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            recovery_window=recovery_window,
            high_low_window=high_low_window,
            sma_window_stock=sma_window_stock,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            low_proximity=low_proximity,
        )
        self.rebalance_every = int(rebalance_every)
        self.recovery_window = int(recovery_window)
        self.high_low_window = int(high_low_window)
        self.sma_window_stock = int(sma_window_stock)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.low_proximity = float(low_proximity)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + self.high_low_window + self.recovery_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Need enough data for 52wk high/low + SMA + recovery window
            need = self.high_low_window + self.recovery_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.high_low_window:
                return []

            scores: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                n = len(col)

                # Need enough history for all indicators
                if n < self.high_low_window:
                    continue

                current_price = float(col.iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    continue

                # 52-week low proximity filter
                window_prices = col.values[-self.high_low_window:]
                low_52wk = float(np.min(window_prices))
                if low_52wk <= 0:
                    continue
                low_ratio = current_price / low_52wk
                # Must be within 20% of 52-week low (ratio <= 1.20)
                if low_ratio > self.low_proximity:
                    continue

                # Per-stock 63d SMA quality gate (structure improving)
                if n < self.sma_window_stock + 2:
                    continue
                sma_63 = float(np.mean(col.values[-self.sma_window_stock:]))
                if sma_63 <= 0:
                    continue
                # Must be above its 63d SMA
                if current_price <= sma_63:
                    continue

                # 21d recovery momentum as ranking criterion
                if n < self.recovery_window + 2:
                    continue
                p_start = float(col.iloc[-self.recovery_window])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                ret_21d = current_price / p_start - 1.0
                if not np.isfinite(ret_21d):
                    continue

                scores[sym] = ret_21d

            if len(scores) < 3:
                # Not enough recovery candidates: IEF defensive
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                # Equal-weight (higher vol names → equal-weight more appropriate)
                weight_per = self.exposure / k
                for sym in ranked:
                    target[sym] = weight_per

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target weights
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
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_52wk_low_recovery"
HYPOTHESIS = (
    "SP500 52-week-low recovery momentum: rank SP500 stocks by 21d return that are "
    "within 20% of their 52-week low (recovery candidates) but above their 63d SMA "
    "(structure improving); hold top-10 equally weighted; SPY 200d gate to IEF; "
    "biweekly rebalance — captures early-stage mean-reversion in beaten-down quality "
    "stocks rather than chasing high-momentum leaders; orthogonal to all existing "
    "momentum strategies"
)

UNIVERSE = _universe

STRATEGY = SP500FiftyTwoWeekLowRecovery()
