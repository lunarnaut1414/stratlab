"""VNQ-TLT Yield-Sensitivity Regime — gen_8 sonnet-7

Hypothesis: Use VNQ/TLT 30d return spread as a REIT-vs-bond yield-sensitivity
signal:
  - VNQ outperforms TLT (REITs leading bonds): rate environment favors
    income-seeking equities = risk-on yield regime → hold top-15 SP500
    momentum stocks
  - TLT outperforms VNQ (bonds outperform REITs): rising-rate stress or
    flight-to-quality → hold TLT 60% + GLD 37%
  - SPY 200d SMA outer gate: bear market always routes to TLT

Rationale:
  REITs (VNQ) are highly rate-sensitive equities that underperform bonds
  when yields spike (rate shock) but outperform bonds in stable-rate bull
  regimes. The VNQ/TLT spread is a market-based proxy for rate regime that
  incorporates real-time credit, duration, and equity sentiment — distinct
  from TNX yield direction or yield-curve slope signals which only measure
  rate levels, not equity sensitivity to those rates.

This should produce a portfolio decorrelated from VIX-gating, JNK-credit,
and yield-curve-slope strategies on the leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # ~3 months for SP500 stock ranking
REGIME_WINDOW = 30         # VNQ/TLT spread window
TREND_WINDOW = 200         # 200d SMA outer gate
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_VNQ = "VNQ"
_GLD = "GLD"


class VNQTLTYieldRegime(Strategy):
    """REIT-vs-bond yield-sensitivity regime router."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        regime_window: int = REGIME_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            regime_window=regime_window,
            trend_window=trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.regime_window = int(regime_window)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
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
        bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            # SPY bear: always TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute VNQ vs TLT 30d return spread
            risk_on = False
            try:
                vnq_hist = ctx.history(_VNQ)
                tlt_hist = ctx.history(_TLT)
                if (vnq_hist is not None and tlt_hist is not None and
                        len(vnq_hist) >= self.regime_window + 2 and
                        len(tlt_hist) >= self.regime_window + 2):
                    vnq_close = vnq_hist["close"].dropna()
                    tlt_close = tlt_hist["close"].dropna()
                    if (len(vnq_close) >= self.regime_window + 1 and
                            len(tlt_close) >= self.regime_window + 1):
                        vnq_ret = float(vnq_close.iloc[-1] / vnq_close.iloc[-self.regime_window] - 1.0)
                        tlt_ret = float(tlt_close.iloc[-1] / tlt_close.iloc[-self.regime_window] - 1.0)
                        if np.isfinite(vnq_ret) and np.isfinite(tlt_ret):
                            risk_on = vnq_ret > tlt_ret
            except Exception:
                pass

            if risk_on:
                # Risk-on: top-K SP500 momentum stocks
                need = self.momentum_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 1:
                    if _TLT in live:
                        target[_TLT] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _VNQ, _GLD):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + 1:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret

                    if len(scores) < 5:
                        if _TLT in live:
                            target[_TLT] = self.exposure
                    else:
                        k = min(self.top_k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                        per_weight = self.exposure / len(ranked)
                        for sym in ranked:
                            if sym in live:
                                target[sym] = per_weight
            else:
                # Bonds outperforming REITs: defensive
                tlt_w = self.exposure * 0.618  # ~60%
                gld_w = self.exposure * 0.382  # ~37% (remainder)
                if _TLT in live:
                    target[_TLT] = tlt_w
                if _GLD in live:
                    target[_GLD] = gld_w

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
    return sp500_tickers() + [_TLT, _SPY, _VNQ, _GLD]


NAME = "vnq_tlt_yield_regime"
HYPOTHESIS = (
    "VNQ-TLT yield-sensitivity regime: use VNQ/TLT 30d return spread as REIT-vs-bond signal; "
    "when VNQ outperforms TLT (low rate regime, risk-on) hold top-15 SP500 stocks by 63d "
    "momentum equal-weight; when TLT outperforms VNQ hold TLT 60% + GLD 37%; SPY 200d SMA "
    "outer gate; biweekly rebalance — REIT as novel yield-regime proxy distinct from "
    "TNX/yield-curve signals"
)

UNIVERSE = _universe

STRATEGY = VNQTLTYieldRegime()
