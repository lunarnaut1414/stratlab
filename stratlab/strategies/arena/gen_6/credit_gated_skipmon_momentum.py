"""Credit-gated skip-month momentum strategy.

Hypothesis: Use JNK/LQD 30d return as credit regime signal.
  - Risk-on (JNK outperforms LQD over 30 days): hold top-15 SP500 stocks
    by 126d-skip-21d (6 month skip 1 month) momentum, inverse-vol weighted.
  - Risk-off (LQD outperforms JNK): hold TLT 50% + GLD 50%.
  Monthly rebalance (every 21 bars).

Rationale:
  The 6-1 month skip-month momentum is the canonical Fama-French factor
  (Jegadeesh & Titman 1993) that avoids short-term reversal. The credit
  spread signal (JNK/LQD relative performance) captures economic cycle
  risk-on/risk-off transitions that are orthogonal to pure price momentum.
  When credit is tightening (HY outperforms IG), equity momentum works best.

Diversification vs leaderboard:
  - xsect_12m_invvol_goldencross: 126d skip 21d + SPY 200d SMA gate + IEF only.
    This uses JNK/LQD credit gate instead (different regime signal), TLT+GLD
    defensive instead of IEF, and 15 stocks vs 20. Daily return path differs.
  - gen5_vix_gated_sp500_momentum: VIX gate, 63d no-skip momentum. Very different.
  - gen5_credit_spread_hyg_lqd: same credit signal but holds JNK/LQD bonds only
    (bond ETFs, not stocks). Completely different equity exposure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 21        # monthly
MOM_LOOKBACK = 126          # 6 months
MOM_SKIP = 21               # skip last 1 month (canonical skip-month)
VOL_WINDOW = 20             # for inverse-vol weights
TOP_K = 15
CREDIT_WINDOW = 30          # JNK/LQD 30d relative return as regime signal
EXPOSURE = 0.97

_JNK = "JNK"
_LQD = "LQD"


class CreditGatedSkipMonMomentum(Strategy):
    """JNK/LQD 30d credit regime + 126d skip-21d SP500 momentum (inverse-vol)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        credit_window: int = CREDIT_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            top_k=top_k,
            credit_window=credit_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.credit_window = int(credit_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + self.vol_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- Credit regime: JNK vs LQD 30d return ---
        risk_on = True  # default to risk-on if data unavailable
        try:
            jnk_hist = ctx.history(_JNK)
            lqd_hist = ctx.history(_LQD)
            if (
                jnk_hist is not None and lqd_hist is not None
                and len(jnk_hist) >= self.credit_window + 1
                and len(lqd_hist) >= self.credit_window + 1
            ):
                jnk_c = jnk_hist["close"].dropna()
                lqd_c = lqd_hist["close"].dropna()
                if len(jnk_c) >= self.credit_window and len(lqd_c) >= self.credit_window:
                    jnk_ret = float(jnk_c.iloc[-1] / jnk_c.iloc[-self.credit_window] - 1.0)
                    lqd_ret = float(lqd_c.iloc[-1] / lqd_c.iloc[-self.credit_window] - 1.0)
                    if np.isfinite(jnk_ret) and np.isfinite(lqd_ret):
                        risk_on = jnk_ret > lqd_ret
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
            # Risk-off: TLT 50% + GLD 50%
            for sym, w in [("TLT", 0.5), ("GLD", 0.5)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Risk-on: 126d skip 21d momentum, inverse-vol weighted
            need = self.mom_lookback + self.mom_skip + self.vol_window + 2
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_lookback + self.mom_skip:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                # Skip non-stock ETFs
                if sym in {"TLT", "GLD", "SHY", "IEF", "AGG", "JNK", "LQD",
                           "SPY", "QQQ", "IWM", "DBC", "SSO", "TQQQ", "GDX",
                           "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLE", "XLB", "XLY"}:
                    continue
                col = prices[sym].dropna()
                total_needed = self.mom_lookback + self.mom_skip
                if len(col) < total_needed + self.vol_window:
                    continue
                # Skip-month momentum: 126d price return, skipping last 21d
                p_end = float(col.iloc[-(self.mom_skip + 1)])
                p_start = float(col.iloc[-(total_needed + 1)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue
                # Inverse vol weighting
                tail = col.iloc[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < self.top_k:
                # Fallback to TLT if not enough qualifying stocks
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:self.top_k]
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "GLD", "JNK", "LQD", "SPY"]


NAME = "credit_gated_skipmon_momentum"
HYPOTHESIS = (
    "SP500 6-month skip-1-month momentum with credit regime gate: use JNK/LQD 30d return "
    "as risk regime; hold top-15 SP500 by 126d-skip-21d momentum (inverse-vol weighted) in "
    "risk-on; hold TLT+GLD 50/50 when credit spreads widening; monthly rebalance; "
    "combines Fama-French skip-month with credit-spread regime."
)

UNIVERSE = _universe

STRATEGY = CreditGatedSkipMonMomentum()
