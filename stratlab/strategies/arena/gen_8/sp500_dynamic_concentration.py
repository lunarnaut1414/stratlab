"""SP500 Dynamic Concentration — gen_8 sonnet-7

Hypothesis: Use QQQ vs SPY 42d relative return as an equity market leadership
signal to dynamically adjust portfolio concentration:

  - QQQ outperforms SPY (42d): Tech/growth regime → hold top-10 SP500 stocks
    by 63d momentum (more concentrated, higher-beta within momentum theme)
  - SPY outperforms QQQ (42d): Value/broad-market regime → hold top-25 SP500
    stocks by 63d momentum (more diversified, lower concentration)
  - Both SPY and QQQ have negative 42d return: risk-off → hold IEF (defensive)
  - SPY below 200d SMA (outer bear gate): always IEF

Rationale:
  When tech is leading (QQQ > SPY), the momentum factor is concentrated in a
  narrow set of high-growth names — concentration in top-10 captures this more
  efficiently. When broad market leads (SPY > QQQ), the opportunity set is
  wider and diversification across top-25 reduces single-stock risk without
  sacrificing momentum premium.

  This is fundamentally different from all existing strategies: instead of a
  binary on/off switch between equity and defensive, we adjust the SHAPE of
  the equity portfolio based on which type of equity market we're in.

Signal: QQQ vs SPY 42d return spread (both tradeable ETFs — signal from their
price history, trades in individual SP500 stocks)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # stock ranking lookback
REGIME_WINDOW = 42         # QQQ vs SPY leadership window
VOL_WINDOW = 21            # for inverse-vol sizing
TREND_WINDOW = 200         # SPY market gate
TOP_K_GROWTH = 10          # concentrated: tech/growth regime
TOP_K_VALUE = 25           # diversified: value/broad regime
EXPOSURE = 0.97
_SPY = "SPY"
_QQQ = "QQQ"
_IEF = "IEF"


class SP500DynamicConcentration(Strategy):
    """Dynamic portfolio concentration based on QQQ vs SPY 42d leadership."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        regime_window: int = REGIME_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k_growth: int = TOP_K_GROWTH,
        top_k_value: int = TOP_K_VALUE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            regime_window=regime_window,
            vol_window=vol_window,
            trend_window=trend_window,
            top_k_growth=top_k_growth,
            top_k_value=top_k_value,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.regime_window = int(regime_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.top_k_growth = int(top_k_growth)
        self.top_k_value = int(top_k_value)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window, self.regime_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY outer trend gate
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
        bull_market = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull_market:
            # Bear market: defensive IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Compute QQQ vs SPY 42d return spread
            qqq_ret = float("nan")
            spy_42d_ret = float("nan")
            try:
                qqq_hist = ctx.history(_QQQ)
                if qqq_hist is not None and len(qqq_hist) >= self.regime_window + 2:
                    qqq_close = qqq_hist["close"].dropna()
                    if len(qqq_close) >= self.regime_window + 1:
                        qqq_ret = float(qqq_close.iloc[-1] / qqq_close.iloc[-self.regime_window] - 1.0)
            except Exception:
                pass

            if len(spy_close) >= self.regime_window + 1:
                spy_42d_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.regime_window] - 1.0)

            # Determine regime
            if (not np.isfinite(qqq_ret) or not np.isfinite(spy_42d_ret) or
                    (qqq_ret < 0 and spy_42d_ret < 0)):
                # Risk-off: both negative or data missing
                if _IEF in live:
                    target[_IEF] = self.exposure
            else:
                # Choose concentration based on leadership
                if np.isfinite(qqq_ret) and qqq_ret > spy_42d_ret:
                    top_k = self.top_k_growth  # concentrated tech regime
                else:
                    top_k = self.top_k_value   # diversified broad regime

                # Rank SP500 stocks by 63d momentum
                need = self.momentum_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 1:
                    if _IEF in live:
                        target[_IEF] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    vols: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _QQQ, _IEF):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + 1:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret

                        daily_rets = col.pct_change().dropna()
                        if len(daily_rets) >= self.vol_window:
                            rv = float(daily_rets.iloc[-self.vol_window:].std())
                            vols[sym] = max(rv, 1e-6)

                    if len(scores) < 5:
                        if _IEF in live:
                            target[_IEF] = self.exposure
                    else:
                        k = min(top_k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                        # Inverse-vol weighting
                        inv_vols = {}
                        for sym in ranked:
                            vol = vols.get(sym, 0.02)
                            inv_vols[sym] = 1.0 / max(vol, 1e-6)
                        total_inv = sum(inv_vols.values())
                        if total_inv <= 0:
                            per_weight = self.exposure / len(ranked)
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = per_weight
                        else:
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = self.exposure * (inv_vols[sym] / total_inv)

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
    return sp500_tickers() + [_IEF, _SPY, _QQQ]


NAME = "sp500_dynamic_concentration"
HYPOTHESIS = (
    "SP500 dynamic concentration: when QQQ outperforms SPY on 42d return (tech/growth "
    "leadership regime), hold top-10 SP500 stocks by 63d momentum (concentrated); when "
    "SPY leads QQQ (broad-market/value regime), hold top-25 SP500 stocks by 63d momentum "
    "(diversified); when both SPY and QQQ have negative 42d return, hold IEF; SPY 200d "
    "SMA outer bear gate; inverse-vol weighted; biweekly rebalance — dynamically adjusts "
    "concentration based on equity market leadership style"
)

UNIVERSE = _universe

STRATEGY = SP500DynamicConcentration()
