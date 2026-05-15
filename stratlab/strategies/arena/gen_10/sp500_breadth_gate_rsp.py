"""SP500 momentum with RSP/SPY breadth quality gate.

Hypothesis:
    Market breadth quality distinguishes sustainable from narrow rallies.
    RSP (equal-weight S&P 500 ETF) vs SPY (cap-weight S&P 500) ratio captures
    whether broad-based participation (RSP leads) or narrow mega-cap leadership
    (SPY leads) drives the market.

    When RSP 21d return > SPY 21d return (breadth-positive regime):
        - Rally is broad-based: stock selection in SP500 is rewarded
        - Hold top-15 SP500 stocks by 63d momentum, inverse-vol weighted

    When SPY 21d return > RSP 21d return (narrow leadership regime):
        - Momentum is concentrated in mega-caps: single-stock selection risky
        - Hold SPY 60% + IEF 37% blend (participate but reduce single-stock risk)

    SPY 200d SMA outer bear gate: full defensive IEF.

    Biweekly rebalance.

Differentiation from leaderboard:
    - All existing macro gates use external cross-asset signals:
      EEM/SPY (EM flow), EFA/SPY (DM flow), VWO/VEA (intl regime),
      DVY/SPY (yield character), TNX/200d (rate trend), JNK/LQD (credit spread)
    - This uses an INTERNAL market signal (RSP vs SPY) — the ratio of equal-weight
      to cap-weight SP500 — which directly measures breadth vs concentration
    - No existing strategy uses RSP as a signal (RSP is a tradeable ETF with
      IS coverage since 2003)
    - Mechanism: regimes the broader market, not macro conditions

Design:
    - Signal: RSP 21d return vs SPY 21d return (rolling spread)
    - Breadth regime: RSP 21d > SPY 21d → stock selection mode
    - Narrow regime: SPY 21d > RSP 21d → SPY+IEF blend
    - Bear regime (SPY < 200d SMA): IEF defensive
    - Stock selection: top-15 SP500 by 63d momentum, inverse-vol weighted
    - Rebalance: every 10 bars
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOMENTUM_WINDOW = 63       # 3-month momentum for stock selection
BREADTH_WINDOW = 21        # RSP vs SPY 21d return comparison
SPY_TREND_WINDOW = 200     # outer bear gate
VOL_WINDOW = 21            # inv-vol weighting
TOP_K = 15                 # stocks to hold in breadth-positive regime
EXPOSURE = 0.97            # max equity exposure
# Narrow regime blend weights
NARROW_SPY_WT = 0.60
NARROW_IEF_WT = 0.37


class SP500BreadthGateRSP(Strategy):
    """SP500 momentum gated by RSP/SPY breadth quality signal.

    Breadth-positive (RSP leads SPY): top-15 stocks by 63d momentum, inv-vol weighted.
    Narrow (SPY leads RSP): SPY 60% + IEF 37%.
    Bear (SPY < 200d SMA): IEF 97%.
    Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        narrow_spy_wt: float = NARROW_SPY_WT,
        narrow_ief_wt: float = NARROW_IEF_WT,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            breadth_window=breadth_window,
            spy_trend_window=spy_trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
            narrow_spy_wt=narrow_spy_wt,
            narrow_ief_wt=narrow_ief_wt,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.breadth_window = int(breadth_window)
        self.spy_trend_window = int(spy_trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.narrow_spy_wt = float(narrow_spy_wt)
        self.narrow_ief_wt = float(narrow_ief_wt)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.spy_trend_window + self.breadth_window + self.momentum_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY outer gate + breadth signal
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + self.breadth_window + 2:
            return []

        # SPY 200d SMA bear gate
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        # RSP breadth signal
        try:
            rsp_hist = ctx.history("RSP")
        except KeyError:
            # If RSP not available, fall back to narrow regime
            rsp_hist = None

        breadth_positive = False
        if rsp_hist is not None:
            rsp_close = rsp_hist["close"].dropna()
            if len(rsp_close) >= self.breadth_window + 2 and len(spy_close) >= self.breadth_window + 2:
                rsp_ret = float(rsp_close.iloc[-1]) / float(rsp_close.iloc[-self.breadth_window]) - 1.0
                spy_ret = float(spy_close.iloc[-1]) / float(spy_close.iloc[-self.breadth_window]) - 1.0
                breadth_positive = rsp_ret > spy_ret

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market: full defensive IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        elif not breadth_positive:
            # Narrow leadership: SPY + IEF blend
            if "SPY" in closes_now.index:
                target["SPY"] = self.narrow_spy_wt
            if "IEF" in closes_now.index:
                target["IEF"] = self.narrow_ief_wt
        else:
            # Breadth-positive: aggressive stock selection
            need = self.momentum_window + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # 63d momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Per-stock 200d SMA check (only above SMA)
                # Use what we have in the closes_window (might be shorter than 200d)
                # Skip this filter if not enough data — rely on outer SPY gate

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Not enough candidates: SPY + IEF blend
                if "SPY" in closes_now.index:
                    target["SPY"] = self.narrow_spy_wt
                if "IEF" in closes_now.index:
                    target["IEF"] = self.narrow_ief_wt
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["RSP", "SPY", "IEF"]


NAME = "sp500_breadth_gate_rsp"
HYPOTHESIS = (
    "SP500 breadth-quality gate: use RSP/SPY 21d return ratio to distinguish "
    "broad-based vs narrow rallies; when RSP leads SPY (breadth positive, RSP 21d > SPY 21d) "
    "hold top-15 SP500 stocks by 63d momentum inverse-vol weighted; when SPY leads RSP "
    "(narrow leadership) hold SPY 60%+IEF 37% blend; SPY 200d bear gate to IEF; "
    "biweekly rebalance — RSP/SPY spread as breadth quality regime filter distinct from "
    "all leaderboard macro gates"
)

UNIVERSE = _universe

STRATEGY = SP500BreadthGateRSP()
