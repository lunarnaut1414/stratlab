"""SP500 momentum gated by RSP-vs-SPY breadth regime — gen_10 sonnet-7

Hypothesis: When RSP (equal-weight SP500) 42d return exceeds SPY 42d return
(broad market participation), stock selection momentum works better because
gains are distributed widely. When SPY leads RSP (mega-cap concentration),
reduce to a SPY+TLT blend. SPY 200d outer bear gate to TLT.

Rationale:
  - RSP outperforming SPY implies broad participation: stocks have similar
    returns across the distribution, so cross-sectional momentum picks work.
  - SPY outperforming RSP implies mega-cap concentration: momentum leaders
    are just the same mega-caps driving SPY. Cross-sectional selection adds
    less value; better to hold SPY itself.
  - This is a selection-quality regime gate, NOT a macro-signal gate.
    It does not depend on VIX level, yield-curve shape, or credit spreads
    (all saturated). Mechanism is orthogonal to IS-window calm-VIX bias.
  - RSP covers full IS window (starts 2003). SPY covers full IS. TLT covers
    full IS. All safe.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
BREADTH_WINDOW = 42       # 42d RSP vs SPY return comparison
MOMENTUM_WINDOW = 126     # 6-month stock momentum
VOL_WINDOW = 21           # inverse-vol weighting lookback
SPY_TREND_WINDOW = 200    # outer bear-market gate
TOP_K = 15
EXPOSURE = 0.97
NEUTRAL_SPY_W = 0.60      # SPY weight in neutral (concentration) regime
NEUTRAL_TLT_W = 0.37      # TLT weight in neutral regime


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["RSP", "SPY", "TLT"]


UNIVERSE = _universe


class RspBreadthRegimeSP500(Strategy):
    """SP500 126d momentum with RSP-vs-SPY breadth regime gate; inverse-vol
    weighted; SPY 200d outer bear gate to TLT; biweekly rebalance.
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
        neutral_spy_w: float = NEUTRAL_SPY_W,
        neutral_tlt_w: float = NEUTRAL_TLT_W,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            momentum_window=momentum_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            neutral_spy_w=neutral_spy_w,
            neutral_tlt_w=neutral_tlt_w,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.neutral_spy_w = float(neutral_spy_w)
        self.neutral_tlt_w = float(neutral_tlt_w)

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
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            # Full bear: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Breadth regime: RSP vs SPY 42d return
            rsp_hist = ctx.history("RSP")
            rsp_ok = len(rsp_hist) >= self.breadth_window + 2
            broad_regime = False  # default to concentration regime

            if rsp_ok:
                rsp_close = rsp_hist["close"].dropna()
                spy_close_b = spy_hist["close"].dropna()
                if len(rsp_close) >= self.breadth_window + 1 and len(spy_close_b) >= self.breadth_window + 1:
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                    spy_ret = float(spy_close_b.iloc[-1] / spy_close_b.iloc[-self.breadth_window] - 1.0)
                    broad_regime = rsp_ret > spy_ret  # RSP leads = broad

            if not broad_regime:
                # Concentration regime: SPY + TLT blend
                if "SPY" in live:
                    target["SPY"] = self.neutral_spy_w
                if "TLT" in live:
                    target["TLT"] = self.neutral_tlt_w
            else:
                # Broad participation: stock momentum
                need = self.momentum_window + self.vol_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < need - 5:
                    # fallback to SPY blend
                    if "SPY" in live:
                        target["SPY"] = self.neutral_spy_w
                    if "TLT" in live:
                        target["TLT"] = self.neutral_tlt_w
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}

                    for sym in prices.columns:
                        if sym in ("RSP", "SPY", "TLT"):
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
                        if "SPY" in live:
                            target["SPY"] = self.neutral_spy_w
                        if "TLT" in live:
                            target["TLT"] = self.neutral_tlt_w
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


NAME = "rsp_breadth_regime_sp500"
HYPOTHESIS = (
    "RSP-vs-SPY breadth regime gate on SP500 momentum: when RSP(equal-weight SP500) 42d return "
    "exceeds SPY 42d return by >0pp (broad market participation), hold top-15 SP500 stocks by "
    "126d momentum; when SPY leads RSP (mega-cap concentration regime), hold SPY 60pct+TLT 37pct; "
    "SPY 200d bear gate to TLT; inverse-vol weighted; biweekly rebalance"
)

STRATEGY = RspBreadthRegimeSP500()
