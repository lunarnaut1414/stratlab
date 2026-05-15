"""gen_9 sonnet-6 — Long-end Slope + Skip-Month Momentum + Inverse-Vol Weighting

Hypothesis: Combine two gen_8 top performers into a single strategy:
  - Macro gate: 30Y-10Y long-end yield slope (TYX-TNX) vs its 200d MA
    (best OOS macro signal from gen_8 — 95% IS retention)
  - Stock selection: 126d-skip-21d skip-month Jegadeesh-Titman momentum
    (best OOS stock selector from gen_8 — 79% IS retention)
  - Position sizing: inverse realized-vol weighting (reduces concentration in
    volatile names vs equal-weighting)

When long-end slope is STEEP (TYX-TNX > 200d MA = term premium expanding,
risk-on): hold top-15 SP500 stocks by 126d-skip-21d momentum, inverse-vol
weighted, above 200d SPY gate.
When long-end slope is FLAT/INVERTED: hold SPY 60%+IEF 37% (de-risk partially).
SPY 200d SMA outer bear gate → TLT 97%.

Rationale: gen8_opus1_longend_slope_equity_gate used pure 63d momentum and got
0.79 OOS Calmar. gen8_sp500_skipmon_63sma_momentum used SPY 200d gate only and
got 0.63 OOS Calmar. Neither combined the two. Skip-month avoids short-term
reversal contamination; long-end slope avoids VIX/credit redundancy; inverse-vol
weighting provides automatic deleveraging for volatile names. The triple
combination should produce a more stable cross-half profile.

Coverage check:
- ^TYX available from 1977 (deep history)
- ^TNX available from 1962
- 200d MA warmup on slope: ~200 bars, handled in warmup
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_LONG = 126         # total lookback for skip-month
MOM_SKIP = 21          # skip recent N days to avoid reversal
SLOPE_TREND = 200      # MA window for long-end slope
SPY_TREND = 200
VOL_WINDOW = 21        # for inverse-vol weighting
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_TYX = "^TYX"
_TNX = "^TNX"


class LongendSlopeSkipmonInvvol(Strategy):
    """SP500 skip-month momentum with long-end slope gate and inverse-vol sizing."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        slope_trend: int = SLOPE_TREND,
        spy_trend: int = SPY_TREND,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            slope_trend=slope_trend,
            spy_trend=spy_trend,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.slope_trend = int(slope_trend)
        self.spy_trend = int(spy_trend)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slope_trend, self.spy_trend, self.mom_long) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- 30Y-10Y long-end slope vs its 200d MA ---
        slope_steep = True  # default risk-on if signal unavailable
        try:
            tyx_hist = ctx.history(_TYX)
            tnx_hist = ctx.history(_TNX)
            if (tyx_hist is not None and tnx_hist is not None
                    and len(tyx_hist) >= self.slope_trend + 2
                    and len(tnx_hist) >= self.slope_trend + 2):
                tyx_close = tyx_hist["close"].dropna()
                tnx_close = tnx_hist["close"].dropna()
                n = min(len(tyx_close), len(tnx_close))
                if n >= self.slope_trend + 1:
                    slope = tyx_close.values[-n:] - tnx_close.values[-n:]
                    slope_ma = float(np.mean(slope[-self.slope_trend:]))
                    slope_now = float(slope[-1])
                    slope_steep = slope_now > slope_ma
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not slope_steep:
            # Flat/inverted long-end slope: SPY+IEF blend
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _IEF in live:
                target[_IEF] = self.exposure * 0.379
        else:
            # Steep slope + SPY bull: skip-month momentum with inverse-vol weighting
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Skip-month momentum: return from mom_long ago to mom_skip ago (skip recent)
                scores: dict[str, float] = {}
                vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue

                    # Skip-month: return from index[-mom_long] to index[-mom_skip]
                    p_start = float(col.iloc[-self.mom_long])
                    p_end = float(col.iloc[-self.mom_skip])
                    if p_start <= 0:
                        continue
                    skip_ret = p_end / p_start - 1.0
                    if not np.isfinite(skip_ret):
                        continue
                    scores[sym] = skip_ret

                    # Realized vol for inverse-vol weighting
                    if len(col) >= self.vol_window + 1:
                        log_rets = np.log(col.values[1:] / col.values[:-1])
                        rv = float(np.std(log_rets[-self.vol_window:]) * np.sqrt(252))
                        if rv > 0 and np.isfinite(rv):
                            vols[sym] = rv

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

                    # Inverse-vol weighting for the selected stocks
                    inv_vols = {}
                    for sym in ranked:
                        if sym in vols and vols[sym] > 0:
                            inv_vols[sym] = 1.0 / vols[sym]
                        else:
                            inv_vols[sym] = 1.0  # equal weight fallback

                    total_inv_vol = sum(inv_vols.values())
                    if total_inv_vol > 0:
                        for sym in ranked:
                            w = inv_vols.get(sym, 1.0) / total_inv_vol * self.exposure
                            if sym in live:
                                target[sym] = w
                    else:
                        per_weight = self.exposure / len(ranked)
                        for sym in ranked:
                            if sym in live:
                                target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY, _IEF, _TYX, _TNX]


NAME = "longend_slope_skipmon_invvol"
HYPOTHESIS = (
    "Long-end slope (TYX-TNX) macro gate with skip-month (126d-skip-21d) stock momentum "
    "+ inverse-vol weighting: when TYX-TNX above its 200d MA (term premium expanding = "
    "risk-on), hold top-15 SP500 stocks by 126d-skip-21d momentum, inverse-vol weighted; "
    "when slope flat/inverted, hold SPY 60%+IEF 37%; SPY 200d outer bear gate to TLT; "
    "biweekly rebalance; combines gen8 best macro signal with gen8 best stock selector"
)

UNIVERSE = _universe

STRATEGY = LongendSlopeSkipmonInvvol()
