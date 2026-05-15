"""SP500 momentum gated by stock-bond realized correlation regime.

Hypothesis: The stock-bond correlation regime is a structurally different macro signal
from VIX level, credit spreads, or yield curve slope. In normal market regimes, equities
and bonds are negatively correlated (flight-to-quality). When they become positively
correlated (both selling off simultaneously), it signals a regime of inflation or
liquidity stress where traditional diversification fails. Use this regime to gate
SP500 momentum stock selection.

When SPY-TLT 42d realized correlation < +0.2 (normal/negative correlation = standard
risk-off dynamics work): hold top-15 SP500 stocks by 126d momentum.
When SPY-TLT 42d realized correlation > +0.2 (positive correlation = correlated
selloff / inflation regime): hold IEF as defensive (duration-neutral).
SPY 200d SMA outer bear gate always active.

Rationale:
  - Stock-bond correlation as regime gate was NOT tried in any prior gen (gens 5-9).
  - It's orthogonal to VIX-level gates, credit-spread gates, and yield-curve-slope gates.
  - When stocks and bonds sell together (2022-style), there is no safe haven in TLT — the
    correct response is to be defensive via short-duration (IEF) or cash-like instruments.
  - The correlation regime is also structurally less timed to the IS window (2010-2018 was
    mostly negative stock-bond corr) than e.g. VIX < 20.

Design:
  - Compute SPY 42d daily returns; compute TLT 42d daily returns.
  - Pearson correlation over those 42 returns.
  - If correlation < correlation_threshold (default 0.2): hold SP500 top-15 by 126d momentum.
  - If correlation >= threshold OR SPY below 200d SMA: hold IEF.
  - Inverse-vol weighted stock selection.
  - Portfolio vol-target (13% ann); biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10           # biweekly
MOMENTUM_WINDOW = 126          # ~6 months
CORR_WINDOW = 42               # SPY-TLT correlation window
CORR_THRESHOLD = 0.2           # positive corr above this = risk-off regime
VOL_WINDOW = 21                # inverse-vol weight
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE_MAX = 0.97
EXPOSURE_MIN = 0.50
VOL_TARGET = 0.13              # 13% annualized portfolio vol target
ANNUAL_FACTOR = 252.0


class SP500StockBondCorrRegime(Strategy):
    """SP500 126d momentum gated by SPY-TLT 42d realized correlation regime;
    inverse-vol weighted; portfolio vol-targeting; SPY 200d outer gate;
    IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        corr_window: int = CORR_WINDOW,
        corr_threshold: float = CORR_THRESHOLD,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_max: float = EXPOSURE_MAX,
        exposure_min: float = EXPOSURE_MIN,
        vol_target: float = VOL_TARGET,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            corr_window=corr_window,
            corr_threshold=corr_threshold,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_max=exposure_max,
            exposure_min=exposure_min,
            vol_target=vol_target,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.corr_window = int(corr_window)
        self.corr_threshold = float(corr_threshold)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_max = float(exposure_max)
        self.exposure_min = float(exposure_min)
        self.vol_target = float(vol_target)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.corr_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY history for trend gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + self.corr_window + 5:
            return []

        # SPY 200d SMA gate
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # TLT history for correlation computation
        try:
            tlt_hist = ctx.history("TLT")
        except KeyError:
            tlt_hist = None

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        # Compute SPY-TLT correlation
        corr_signal = -1.0  # default: negative correlation = normal regime
        if tlt_hist is not None and len(tlt_hist) >= self.corr_window + 2:
            tlt_close = tlt_hist["close"].dropna()
            if len(tlt_close) >= self.corr_window + 2 and len(spy_close) >= self.corr_window + 2:
                spy_rets = np.diff(np.log(spy_close.values[-(self.corr_window + 2):] + 1e-10))
                tlt_rets = np.diff(np.log(tlt_close.values[-(self.corr_window + 2):] + 1e-10))
                n = min(len(spy_rets), len(tlt_rets))
                if n >= 20:
                    s_std = float(np.std(spy_rets[-n:]))
                    t_std = float(np.std(tlt_rets[-n:]))
                    if s_std > 1e-8 and t_std > 1e-8:
                        corr_signal = float(np.corrcoef(spy_rets[-n:], tlt_rets[-n:])[0, 1])

        # Determine regime: defensive when SPY bearish OR positive stock-bond correlation
        defensive = (not spy_bull) or (corr_signal >= self.corr_threshold)

        if defensive:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure_max
        else:
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 10:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                if sym in ("SPY", "TLT", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < need - 10:
                    continue

                arr = col.values

                # 126d momentum
                if len(arr) < self.momentum_window + 2:
                    continue
                p_end = float(arr[-1])
                p_start = float(arr[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                if len(arr) < self.vol_window + 1:
                    continue
                tail = arr[-(self.vol_window + 1):]
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure_max
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []

                raw_weights = {sym: self.exposure_max * inv_vols[sym] / iv_sum for sym in ranked}

                # Portfolio vol-targeting proxy
                port_daily_vol = sum(
                    raw_weights[sym] * (1.0 / inv_vols[sym]) for sym in ranked
                )
                port_ann_vol = port_daily_vol * (ANNUAL_FACTOR ** 0.5)
                if port_ann_vol > 1e-6:
                    scale = self.vol_target / port_ann_vol
                    scale = float(np.clip(scale, self.exposure_min / self.exposure_max, 1.0))
                else:
                    scale = 1.0

                for sym in ranked:
                    target[sym] = raw_weights[sym] * scale

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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
    return sp500_tickers() + ["IEF", "TLT", "SPY"]


NAME = "sp500_stock_bond_corr_regime"
HYPOTHESIS = (
    "SP500 top-15 momentum (126d) gated by SPY-TLT 42d realized correlation regime: hold stocks "
    "when correlation < +0.2 (normal negative stock-bond relationship); hold IEF when correlation "
    ">= +0.2 (inflation/liquidity stress where stocks and bonds sell together); SPY 200d outer "
    "bear gate also applies; inverse-vol weighted; portfolio vol-target (13% ann); biweekly "
    "rebalance — stock-bond correlation regime is orthogonal to VIX level, credit spread, and "
    "yield curve slope signals used in existing strategies"
)

UNIVERSE = _universe

STRATEGY = SP500StockBondCorrRegime()
