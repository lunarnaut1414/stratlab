"""opus-1 mutation of gen10_rsp_breadth_regime_sp500 (IS 1.38, h1=1.43/h2=1.55).

Parent: stratlab/strategies/arena/gen_10/rsp_breadth_regime_sp500.py

Hypothesis (opus-1, gen_10):
    The gen_8 OOS lesson — "defensive-branch divergence drives OOS retention,
    not risk-on similarity" — points to mutating the defensive sleeve of the
    top performer rather than its alpha signal. Parent uses (a) SPY 60pct +
    TLT 37pct in concentration regime, (b) TLT 97pct in SPY-bear, both of
    which depend on bond duration carrying the defensive load.

    This variant replaces BOTH defensive branches with SHY (1-3y Treasuries,
    near-cash). SHY has near-zero duration and effectively no rate-cycle
    sensitivity, so its OOS performance is uncorrelated with TLT/IEF cycle
    timing.  When the regime gates fire, the strategy goes to a defensive
    sleeve that simply stops bleeding instead of taking duration bets that
    may not work outside the IS calm-VIX 2010-2018 window.

    Risk-on branch (broad-participation regime): unchanged top-15 SP500 126d
    momentum, inverse-vol weighted, 0.97 gross.

    The gen_8 longend_slope_equity_gate result (95pct retention via defensive-
    branch swap) is the template: keep the working alpha signal, change ONLY
    the defensive sleeve, gain OOS diversity.

Diversification rationale:
    - Parent corr 0.774 to top-5. SHY (instead of SPY+TLT/TLT) has very
      different daily returns than parent during defensive engagements,
      which should keep us under 0.85 corr while preserving identical
      risk-on signal.
    - In IS calm-VIX-tilt windows, SHY contributes ~0% over defensive periods
      vs parent's TLT rally; this is a small IS Calmar cost but a large OOS
      retention upside if 2018-2024 rate cycles invalidate TLT's defensive
      duration carry.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
BREADTH_WINDOW = 42         # 42d RSP vs SPY return comparison
MOMENTUM_WINDOW = 126       # 6-month stock momentum
VOL_WINDOW = 21             # inverse-vol weighting lookback
SPY_TREND_WINDOW = 200      # outer bear-market gate
TOP_K = 15
EXPOSURE = 0.97
DEFENSIVE_W = 0.97          # SHY exposure when defensive


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["RSP", "SPY", "SHY"]


UNIVERSE = _universe


class Opus1RspBreadthShyDefensive(Strategy):
    """SP500 126d momentum gated by RSP-vs-SPY 42d breadth; defensive sleeve
    is SHY-only (near-cash) instead of SPY+TLT blend; SPY 200d bear gate
    also routes to SHY; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_window: int = BREADTH_WINDOW,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        defensive_w: float = DEFENSIVE_W,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            defensive_w=defensive_w,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.defensive_w = float(defensive_w)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.breadth_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d outer bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        def _go_defensive() -> None:
            if "SHY" in live:
                target["SHY"] = self.defensive_w

        if not spy_bull:
            _go_defensive()
        else:
            # Breadth regime: RSP vs SPY 42d return
            try:
                rsp_hist = ctx.history("RSP")
            except KeyError:
                rsp_hist = None

            broad_regime = False
            if rsp_hist is not None:
                rsp_close = rsp_hist["close"].dropna()
                if (len(rsp_close) >= self.breadth_window + 1
                        and len(spy_close) >= self.breadth_window + 1):
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.breadth_window] - 1.0)
                    broad_regime = rsp_ret > spy_ret

            if not broad_regime:
                _go_defensive()
            else:
                need = self.momentum_window + self.vol_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < need - 5:
                    _go_defensive()
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}

                    for sym in prices.columns:
                        if sym in ("RSP", "SPY", "SHY"):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + 2:
                            continue
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-self.momentum_window])
                        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                            continue
                        ret = p_end / p_start - 1.0
                        if not np.isfinite(ret):
                            continue

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
                        _go_defensive()
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


NAME = "opus1_rsp_breadth_shy_defensive"
HYPOTHESIS = (
    "opus-1 mutation of gen10_rsp_breadth_regime_sp500 (IS 1.38): defensive sleeve in "
    "both concentration-regime and SPY-bear branches replaced with SHY 97pct (near-cash, "
    "zero-duration), instead of parent's SPY+TLT blend / TLT-only; risk-on branch unchanged "
    "(top-15 SP500 126d momentum, inverse-vol, biweekly); defensive-branch divergence per "
    "gen_8 OOS lesson — preserves working alpha signal while diversifying the defensive sleeve "
    "from bond-duration carry"
)

STRATEGY = Opus1RspBreadthShyDefensive()
