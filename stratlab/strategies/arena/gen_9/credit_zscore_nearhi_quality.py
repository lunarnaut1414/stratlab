"""gen_9 sonnet-3 — JNK/LQD Credit Z-Score Gating Near-52w-High Quality Momentum

Hypothesis: Apply the proven JNK/LQD 90d z-score 3-tier credit gate to the proven
near-52w-high quality selection (not standard 63d momentum).

Credit gate (JNK/LQD ratio z-score over 90d window):
  - Strong credit (z > +0.5): top-15 SP500 by 126d momentum AND price>80% of 252d high
    (nearhi quality filter = sustained uptrend stocks)
  - Neutral credit (z -0.5 to +0.5): SPY 60% + IEF 37% (defensive blend)
  - Weak credit (z < -0.5): TLT 97% (full defensive)
  - SPY 200d outer bear gate: -> TLT 97%
  - Inverse-vol weighting in equity mode
  - Biweekly rebalance

Rationale:
  - gen8_sp500_credit_zscore_3tier (IS Calmar 0.88, OOS 0.40) uses same z-score gate
    but selects by standard 63d momentum
  - gen6_nearhi_momentum_quality (IS Calmar 1.16, OOS 0.63) uses nearhi quality filter
    but gates by simple SPY 200d SMA only
  - COMBINATION: credit z-score gate (more sophisticated than SPY SMA) + nearhi quality
    (more selective than standard momentum) = novel combination absent from leaderboard

Distinction from existing:
  - Different from gen8_sp500_credit_zscore_3tier: uses nearhi 126d quality selection vs
    standard 63d momentum (different stock set selected)
  - Different from gen6_nearhi_momentum_quality: credit z-score gate vs simple SPY 200d
    (more precise credit regime detection)
  - Different from gen8_opus1_sp500_credit_zscore_qqq_neutral: same gate but different
    neutral-tier (QQQ) vs my neutral-tier (SPY+IEF) — also different equity selection
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOM_WINDOW = 63             # 63d momentum (shorter, more responsive than 126d)
HIGH_WINDOW = 252           # 52-week high lookback
NEARHI_THRESHOLD = 0.80     # price must be > 80% of 252d high (quality gate)
ZSCORE_WINDOW = 90          # JNK/LQD z-score lookback
ZSCORE_HIGH = 0.5           # strong credit threshold
ZSCORE_LOW = -0.5           # weak credit threshold
TREND_WINDOW = 200          # SPY 200d SMA
VOL_WINDOW = 21
TOP_K = 15
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_JNK = "JNK"
_LQD = "LQD"


class CreditZscoreNearhiQuality(Strategy):
    """JNK/LQD 90d z-score 3-tier gate applied to nearhi-quality SP500 momentum selection."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        high_window: int = HIGH_WINDOW,
        nearhi_threshold: float = NEARHI_THRESHOLD,
        zscore_window: int = ZSCORE_WINDOW,
        zscore_high: float = ZSCORE_HIGH,
        zscore_low: float = ZSCORE_LOW,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            high_window=high_window,
            nearhi_threshold=nearhi_threshold,
            zscore_window=zscore_window,
            zscore_high=zscore_high,
            zscore_low=zscore_low,
            trend_window=trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.high_window = int(high_window)
        self.nearhi_threshold = float(nearhi_threshold)
        self.zscore_window = int(zscore_window)
        self.zscore_high = float(zscore_high)
        self.zscore_low = float(zscore_low)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def _credit_zscore(self, ctx: BarContext) -> float:
        """Compute JNK/LQD ratio 90d z-score. Returns 0.0 if unavailable."""
        try:
            jnk_hist = ctx.history(_JNK)
            lqd_hist = ctx.history(_LQD)
            if (jnk_hist is None or lqd_hist is None or
                    len(jnk_hist) < self.zscore_window + 5 or
                    len(lqd_hist) < self.zscore_window + 5):
                return 0.0
            jnk_close = jnk_hist["close"].dropna()
            lqd_close = lqd_hist["close"].dropna()
            n = min(len(jnk_close), len(lqd_close))
            if n < self.zscore_window + 1:
                return 0.0
            jnk_arr = jnk_close.values[-n:]
            lqd_arr = lqd_close.values[-n:]
            ratio = jnk_arr / lqd_arr
            window_ratio = ratio[-self.zscore_window:]
            mean_r = float(np.mean(window_ratio))
            std_r = float(np.std(window_ratio))
            if std_r <= 0:
                return 0.0
            zscore = (float(ratio[-1]) - mean_r) / std_r
            return float(zscore) if np.isfinite(zscore) else 0.0
        except Exception:
            return 0.0

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.high_window,
                     self.mom_window, self.zscore_window) + 10
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

        # --- SPY 200d outer bear gate ---
        spy_bull = True
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_now = float(spy_close.iloc[-1])
                    spy_bull = spy_now > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute credit z-score
            z = self._credit_zscore(ctx)

            if z < self.zscore_low:
                # Weak credit: fully defensive TLT
                if _TLT in live:
                    target[_TLT] = self.exposure

            elif z < self.zscore_high:
                # Neutral credit: SPY + IEF blend
                if _SPY in live:
                    target[_SPY] = self.exposure * 0.618
                if _IEF in live:
                    target[_IEF] = self.exposure * 0.382

            else:
                # Strong credit: nearhi quality momentum selection
                need = max(self.mom_window, self.high_window) + 10
                prices = ctx.closes_window(need)
                if len(prices) < self.mom_window:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    quality_scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _IEF, _JNK, _LQD):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.mom_window + 5:
                            continue

                        p_now = float(col.iloc[-1])
                        if p_now <= 0:
                            continue

                        # 126d momentum
                        p_126 = float(col.iloc[-self.mom_window])
                        if p_126 <= 0:
                            continue
                        mom_ret = p_now / p_126 - 1.0
                        if not np.isfinite(mom_ret) or mom_ret <= 0:
                            continue

                        # Near-52w-high quality filter
                        if len(col) >= self.high_window:
                            hi_252 = float(col.iloc[-self.high_window:].max())
                        else:
                            hi_252 = float(col.max())
                        if hi_252 <= 0:
                            continue
                        nearhi = p_now / hi_252
                        if nearhi < self.nearhi_threshold:
                            continue  # not near high

                        if sym in live:
                            quality_scores[sym] = mom_ret

                    if len(quality_scores) < 5:
                        # Fallback to SPY blend if not enough nearhi stocks
                        if _SPY in live:
                            target[_SPY] = self.exposure * 0.618
                        if _IEF in live:
                            target[_IEF] = self.exposure * 0.382
                    else:
                        ranked = sorted(quality_scores, key=quality_scores.__getitem__, reverse=True)
                        candidates = ranked[:self.top_k]

                        # Inverse-vol weighting
                        inv_vols: dict[str, float] = {}
                        for sym in candidates:
                            try:
                                s_hist = ctx.history(sym)
                                if s_hist is None:
                                    continue
                                s_close = s_hist["close"].dropna()
                                if len(s_close) >= self.vol_window + 1:
                                    rets = s_close.iloc[-(self.vol_window + 1):].pct_change().dropna()
                                    rv = float(rets.std()) * np.sqrt(252)
                                else:
                                    rv = 0.20
                            except Exception:
                                rv = 0.20
                            if rv <= 0:
                                rv = 0.20
                            if sym in live:
                                inv_vols[sym] = 1.0 / rv

                        if not inv_vols:
                            if _SPY in live:
                                target[_SPY] = self.exposure
                        else:
                            total = sum(inv_vols.values())
                            for sym, iv in inv_vols.items():
                                target[sym] = self.exposure * iv / total

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_TLT, _IEF, _SPY, _JNK, _LQD]


UNIVERSE = _universe

NAME = "credit_zscore_nearhi_quality"
HYPOTHESIS = (
    "JNK/LQD 90d z-score 3-tier gate on SP500 nearhi-quality momentum: "
    "z>+0.5 hold top-15 SP500 stocks by 126d momentum AND price>80% of 252d high "
    "(nearhi quality filter) 97%; z-0.5 to +0.5 hold SPY 60%+IEF 37% (neutral); "
    "z<-0.5 hold TLT 97%; SPY 200d bear gate; inverse-vol weighted; biweekly; "
    "combines proven credit z-score gating (gen8 OOS 0.40) with proven nearhi quality selection "
    "(gen6 OOS 0.63)"
)

STRATEGY = CreditZscoreNearhiQuality()
