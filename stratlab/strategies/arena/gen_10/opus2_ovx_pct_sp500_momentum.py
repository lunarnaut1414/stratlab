"""^OVX (Crude-Oil VIX) 252d-percentile gate on SP500 momentum.

Hypothesis (opus-2, gen_10 gap_finder):
    ^OVX (CBOE Oil Volatility Index) 252d percentile gates allocation:
    - Low OVX pct (<40th): hold top-15 SP500 stocks by 126d-skip-21d momentum
      (inverse-vol weighted) — calm oil vol == benign commodity-stress regime.
    - High OVX pct (>75th): hold TLT 60% + IEF 37% — elevated oil vol historically
      coincides with energy-supply / geopolitical / inflation stress, where
      duration + short-end blend out-performs equity beta.
    - Mid OVX pct (40-75): hold SPY 97% — neutral.
    SPY 200d outer bear gate to TLT.  Biweekly rebalance.

Why this is an OPEN frontier (phase2_brief gap):
  - ^OVX has NEVER been used as a strategy signal in arena gens 5-10. Only
    ^VIX (equity vol), ^MOVE (rates vol), ^VVIX, ^GVZ, ^SKEW have been touched.
  - Oil vol is structurally orthogonal to equity vol and rates vol — many
    historical commodity-stress episodes (2014-16 oil crash, 2022 supply
    shock) had ^OVX > 50th pct while ^VIX was modest, and vice-versa during
    pure equity panics (2018-Q4, 2020-COVID).
  - The 252d percentile self-calibrates to changing absolute OVX baselines
    (post-shale, post-COVID), avoiding the hard-level overfitting risk.
  - Defensive blend = TLT+IEF (not pure TLT) because oil-driven stress often
    couples with rate volatility — IEF dilutes duration risk vs full TLT.

Distinct from:
  - gen10_move_pct_sp500_momentum: uses ^MOVE (rates vol), same SP500-momentum
    risk-on branch — the gate axis is the differentiator.
  - gen10_sp500_*_momentum quality-filter variants: NO regime gate.
  - gen8 ^GVZ gold-vol regime: uses gold vol on QQQ allocation, different axis.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
OVX_WINDOW = 252
OVX_LOW_PCT = 0.40
OVX_HIGH_PCT = 0.75
MOM_LOOKBACK = 126
MOM_SKIP = 21
VOL_WINDOW = 21
SPY_TREND_WINDOW = 200
TOP_K = 15
EXPOSURE = 0.97
W_TLT_HIGH = 0.60
W_IEF_HIGH = 0.37


class OvxPctSP500Momentum(Strategy):
    """^OVX 252d percentile gate on SP500 momentum cross-section."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ovx_window: int = OVX_WINDOW,
        ovx_low_pct: float = OVX_LOW_PCT,
        ovx_high_pct: float = OVX_HIGH_PCT,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ovx_window=ovx_window,
            ovx_low_pct=ovx_low_pct,
            ovx_high_pct=ovx_high_pct,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.ovx_window = int(ovx_window)
        self.ovx_low_pct = float(ovx_low_pct)
        self.ovx_high_pct = float(ovx_high_pct)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.ovx_window, self.mom_lookback + self.mom_skip) + self.vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
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

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # ^OVX 252d percentile
            ovx_pct = 0.5
            try:
                ovx_hist = ctx.history("^OVX")
                ovx_close = ovx_hist["close"].dropna()
                if len(ovx_close) >= self.ovx_window + 2:
                    cur = float(ovx_close.iloc[-1])
                    window_vals = ovx_close.iloc[-self.ovx_window:].values
                    ovx_pct = float(np.mean(window_vals <= cur))
            except KeyError:
                pass

            if ovx_pct > self.ovx_high_pct:
                if "TLT" in closes_now.index:
                    target["TLT"] = W_TLT_HIGH
                if "IEF" in closes_now.index:
                    target["IEF"] = W_IEF_HIGH
            elif ovx_pct < self.ovx_low_pct:
                # SP500 momentum stock selection
                need = self.mom_lookback + self.mom_skip + self.vol_window + 2
                prices = ctx.closes_window(need)
                if len(prices) < need - 2:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}
                    for sym in prices.columns:
                        col = prices[sym].dropna()
                        if len(col) < self.mom_lookback + self.mom_skip:
                            continue
                        p_end = float(col.iloc[-self.mom_skip - 1])
                        p_start = float(col.iloc[-(self.mom_lookback + self.mom_skip)])
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
                        if "SPY" in closes_now.index:
                            target["SPY"] = self.exposure
                    else:
                        k = min(self.top_k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                        iv_sum = sum(inv_vols[s] for s in ranked)
                        if iv_sum <= 0:
                            return []
                        for sym in ranked:
                            target[sym] = self.exposure * inv_vols[sym] / iv_sum
            else:
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure

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
    return sp500_tickers() + ["SPY", "TLT", "IEF", "^OVX"]


NAME = "opus2_ovx_pct_sp500_momentum"
HYPOTHESIS = (
    "^OVX (oil VIX) 252d percentile gates SP500 momentum: low OVX pct (<40th) hold top-15 SP500 by "
    "126d-skip-21d momentum (inverse-vol weighted); high OVX pct (>75th) hold TLT 60pct+IEF 37pct "
    "defensive; mid hold SPY 97pct; SPY 200d outer bear gate to TLT; biweekly rebalance — "
    "OVX percentile is commodity-vol stress signal orthogonal to VIX (equity) and MOVE (rates)"
)

UNIVERSE = _universe

STRATEGY = OvxPctSP500Momentum()
