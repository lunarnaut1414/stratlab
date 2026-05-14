"""SP500 momentum with short-term reversal filter and JNK credit gate.

Hypothesis: rank top-20 SP500 stocks by 126d return, skip any stock with
a negative 5d return (short-term reversal filter avoids chasing stocks that
just rolled over), inverse-vol weighted; JNK 20d/60d MA crossover as
credit regime gate; TLT defensive when credit risk-off; bi-weekly rebalance.

Rationale: Combining 6-month intermediate momentum with a short-term
reversal filter (remove stocks that just declined over 5 days) helps avoid
catching falling knives at rebalance time — it ensures selected stocks
still have positive recent price action, not just a stale 6-month signal.
The JNK credit gate adds macro regime awareness: when credit spreads are
tightening (JNK 20d MA > 60d MA), the environment is risk-on; when widening,
rotate defensively to TLT. This combination of multi-horizon momentum
filtering with credit gate is distinct from:
  - nearhi_momentum_quality (uses 52w-high proximity, not reversal filter)
  - vix_gated_sp500_momentum (VIX gate vs JNK credit gate)
  - credit_spy_shy_momentum (different momentum window, no reversal filter)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # bars (~2 weeks)
MOMENTUM_WINDOW = 126     # 6-month momentum
REVERSAL_WINDOW = 5       # short-term reversal filter: skip stocks with -5d return
VOL_WINDOW = 20           # for inverse-vol weights
JNK_FAST = 20             # credit fast MA
JNK_SLOW = 60             # credit slow MA
TOP_K = 20
EXPOSURE = 0.97


class SP500MomReversalFilterCredit(Strategy):
    """Top-20 SP500 by 126d momentum, skip negative 5d, inverse-vol sized; JNK credit gate."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        reversal_window: int = REVERSAL_WINDOW,
        vol_window: int = VOL_WINDOW,
        jnk_fast: int = JNK_FAST,
        jnk_slow: int = JNK_SLOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            reversal_window=reversal_window,
            vol_window=vol_window,
            jnk_fast=jnk_fast,
            jnk_slow=jnk_slow,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.reversal_window = int(reversal_window)
        self.vol_window = int(vol_window)
        self.jnk_fast = int(jnk_fast)
        self.jnk_slow = int(jnk_slow)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.jnk_slow) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # JNK credit regime check
        try:
            jnk_hist = ctx.history("JNK")
        except Exception:
            jnk_hist = None

        credit_risk_on = True  # default to risk-on if JNK unavailable
        if jnk_hist is not None and len(jnk_hist) >= self.jnk_slow + 5:
            jnk_close = jnk_hist["close"].dropna()
            if len(jnk_close) >= self.jnk_slow:
                jnk_fast_ma = float(jnk_close.iloc[-self.jnk_fast:].mean())
                jnk_slow_ma = float(jnk_close.iloc[-self.jnk_slow:].mean())
                credit_risk_on = jnk_fast_ma > jnk_slow_ma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not credit_risk_on:
            # Defensive: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Risk-on: top-K SP500 momentum with reversal filter
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.momentum_window + 2:
                    continue

                # Short-term reversal filter: skip stocks with negative 5d return
                if len(col) >= self.reversal_window + 1:
                    recent_ret = float(col.iloc[-1] / col.iloc[-self.reversal_window] - 1.0)
                    if recent_ret < 0:
                        continue  # Skip stocks that just pulled back

                # 126d intermediate momentum
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret) or ret <= 0:
                    continue  # Only hold stocks with positive momentum

                # Inverse-vol weighting
                tail = col.iloc[-self.vol_window - 1:]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail.values[1:] / tail.values[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                # Not enough candidates: fall back to TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

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
    return sp500_tickers() + ["TLT", "JNK", "SPY"]


NAME = "sp500_mom_reversal_filter_credit"
HYPOTHESIS = (
    "SP500 momentum with short-term reversal avoidance: rank top-20 SP500 stocks by 126d return, "
    "skip any stock with negative 5d return (short-term reversal filter), inverse-vol weighted; "
    "JNK 20d/60d MA credit regime gate; TLT defensive; bi-weekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SP500MomReversalFilterCredit()
