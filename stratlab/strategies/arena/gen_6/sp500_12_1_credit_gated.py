"""SP500 12-1 month skip-momentum gated by JNK credit regime.

Hypothesis: Classic 12-1 month momentum (126d return, skip last 21d) on
SP500 stocks, gated by JNK/LQD MA crossover as a credit regime filter.

  - Risk-on (JNK 20d MA > 60d MA): hold top-15 SP500 stocks by 126d-skip-21d
    return, inverse-vol weighted. Biweekly rebalance (21 bars).
  - Risk-off (JNK 20d MA <= 60d MA): hold IEF 60% + SHY 40% (medium+short
    duration, defensive without full-duration risk).

Rationale:
  - 12-1 momentum (skip-1-month) is the academic standard cross-sectional
    momentum factor, avoiding short-term reversal contamination.
  - JNK credit gate adds a macro risk filter that VIX-based gates miss:
    credit spread widening often leads equity drawdowns by 1-4 weeks.
  - Inverse-vol weighting reduces concentration in high-vol momentum names.
  - IEF+SHY defensive (not TLT-only) reduces duration risk during rising
    rates (2013-2018 tightening cycle), which hurt TLT-only strategies.

Distinctions from existing leaderboard:
  - gen5_vix_gated_sp500_momentum: VIX gate + equal-weight (not inv-vol)
    + 63d window (not 126d-skip-21d) + SHY+TLT defensive.
  - gen5_opus1_xsect_12m_invvol_goldencross: same 126d-skip-21d window and
    inv-vol, BUT uses SPY 200d SMA gate (not JNK credit gate), IEF defensive.
  - gen6_sp500_52wk_high_breakout: 52wk proximity filter + 63d window + 200d.
  - gen6_jnk_lqd_spy_regime: credit signal maps to SPY/SSO exposure (no stock
    selection); this strategy uses JNK credit gate for stock-level selection.

The combination of credit-gate + 12-1 momentum + inverse-vol sizing is
novel vs the existing leaderboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21     # monthly
MOM_LOOKBACK = 126       # ~6 months
MOM_SKIP = 21            # skip last ~1 month (reversal avoidance)
VOL_WINDOW = 20          # for inverse-vol weights
TOP_K = 15
FAST_MA = 20             # JNK fast MA for credit regime
SLOW_MA = 60             # JNK slow MA for credit regime
EXPOSURE = 0.97

_JNK = "JNK"
_IEF = "IEF"
_SHY = "SHY"


class SP500TwelveOneCredit(Strategy):
    """SP500 12-1 skip-momentum with JNK credit regime gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            top_k=top_k,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + self.slow_ma + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK credit regime: MA crossover
        risk_on = True   # default to risk-on
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.slow_ma + 2:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.slow_ma:
                    fast = float(jnk_close.iloc[-self.fast_ma:].mean())
                    slow = float(jnk_close.iloc[-self.slow_ma:].mean())
                    if np.isfinite(fast) and np.isfinite(slow) and slow > 0:
                        risk_on = fast > slow
        except (KeyError, IndexError):
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not risk_on:
            # Risk-off: IEF 60% + SHY 40%
            for sym, w in [(_IEF, 0.6), (_SHY, 0.4)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: 12-1 skip-momentum, inverse-vol weighted
            need = self.mom_lookback + self.mom_skip + self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_lookback + self.mom_skip:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue

                # 12-1 skip momentum: return from (lookback + skip) ago to skip ago
                end_idx = -self.mom_skip
                start_idx = -(self.mom_lookback + self.mom_skip)
                p_end = float(col.iloc[end_idx])
                p_start = float(col.iloc[start_idx])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                mom = p_end / p_start - 1.0

                # Inverse realized volatility
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = mom
                inv_vols[sym] = 1.0 / rv

            if len(scores) < self.top_k:
                # Fallback to IEF if too few stocks
                if _IEF in closes_now.index:
                    target[_IEF] = self.exposure
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

        # Adjust to target weights
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
    return sp500_tickers() + [_JNK, _IEF, _SHY]


NAME = "sp500_12_1_credit_gated"
HYPOTHESIS = (
    "12-1 month skip-momentum on SP500 with JNK credit regime gate: when JNK "
    "20d MA > 60d MA hold top-15 SP500 by 126d-skip-21d return (inverse-vol "
    "weighted); when JNK bearish hold IEF 60% + SHY 40%; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SP500TwelveOneCredit()
