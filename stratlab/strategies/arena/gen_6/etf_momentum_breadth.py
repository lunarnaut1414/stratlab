"""ETF momentum with breadth-confirmation filter — gen_6 sonnet-7

Hypothesis: Rank 8 diversified ETFs (QQQ, SPY, IWM, RSP, VGT, XLV, TLT, GLD)
by composite 1m+3m momentum score. Hold top-2 with positive absolute momentum.
Add breadth gate: only hold equity ETFs when RSP/SPY 20d return spread > -2%
(broad market participation; avoid narrow mega-cap-only rallies). Hold IEF if
breadth fails. Monthly rebalance (21 bars).

Rationale:
  RSP (equal-weight SP500) vs SPY (cap-weight) spread is a breadth proxy —
  when RSP underperforms SPY significantly, the rally is narrow (only mega-caps)
  and momentum is more fragile. Combining momentum ranking with breadth filter
  reduces exposure to narrow-breadth tops. This is different from the gen5
  RSP/SPY breadth strategy (gen5_atr_momentum_etf) which directly used RSP/SPY
  as a regime toggle for QQQ vs SPY — here breadth is a secondary filter on
  a multi-ETF momentum ranking.

Distinct from existing leaderboard:
  - RSP/SPY breadth as secondary filter (novel combination)
  - 8-ETF ranking includes both equity styles AND defense in one universe
  - Composite 1m+3m score (not single-lookback)
  - Absolute momentum gate prevents holding declining ETFs
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Diversified ETF universe: growth, broad, small, equal-weight, tech, healthcare, bonds, gold
ETFS = ["QQQ", "SPY", "IWM", "RSP", "VGT", "XLV", "TLT", "GLD"]
CASH_ETF = "IEF"

SHORT_WINDOW = 21     # 1-month return
LONG_WINDOW = 63      # 3-month return
BREADTH_WINDOW = 20   # RSP/SPY spread lookback
BREADTH_MIN = -0.02   # RSP must not lag SPY by more than 2% over 20d
REBALANCE_EVERY = 21  # monthly
TOP_K = 2
EXPOSURE = 0.97


class ETFMomentumBreadth(Strategy):
    """Multi-ETF composite momentum with RSP/SPY breadth confirmation."""

    def __init__(
        self,
        short_window: int = SHORT_WINDOW,
        long_window: int = LONG_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        breadth_min: float = BREADTH_MIN,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            short_window=short_window,
            long_window=long_window,
            breadth_window=breadth_window,
            breadth_min=breadth_min,
            rebalance_every=rebalance_every,
            top_k=top_k,
            exposure=exposure,
        )
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.breadth_window = int(breadth_window)
        self.breadth_min = float(breadth_min)
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def _score(self, hist_close: "np.ndarray") -> float | None:
        if len(hist_close) < self.long_window + 2:
            return None
        short_ret = float(hist_close[-1] / hist_close[-self.short_window] - 1.0)
        long_ret = float(hist_close[-1] / hist_close[-self.long_window] - 1.0)
        if not np.isfinite(short_ret) or not np.isfinite(long_ret):
            return None
        # Equal-weight composite
        return 0.5 * short_ret + 0.5 * long_ret

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.long_window + self.breadth_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- Breadth signal: RSP vs SPY spread ---
        breadth_ok = True  # assume ok if data missing
        try:
            rsp_hist = ctx.history("RSP")
            spy_hist = ctx.history("SPY")
            if len(rsp_hist) >= self.breadth_window + 2 and len(spy_hist) >= self.breadth_window + 2:
                rsp_close = rsp_hist["close"].dropna().values
                spy_close = spy_hist["close"].dropna().values
                rsp_ret = float(rsp_close[-1] / rsp_close[-self.breadth_window] - 1.0)
                spy_ret = float(spy_close[-1] / spy_close[-self.breadth_window] - 1.0)
                spread = rsp_ret - spy_ret
                breadth_ok = spread > self.breadth_min
        except Exception:
            pass

        # --- Score each ETF ---
        scores: dict[str, float] = {}
        for sym in ETFS:
            try:
                hist = ctx.history(sym)
            except Exception:
                continue
            if len(hist) < self.long_window + 2:
                continue
            closes_arr = hist["close"].dropna().values
            score = self._score(closes_arr)
            if score is not None and np.isfinite(score):
                scores[sym] = score

        target: dict[str, float] = {}

        # Absolute momentum gate: only hold ETFs with positive composite score
        positive = {s: r for s, r in scores.items() if r > 0}

        if not positive:
            # All negative: hold cash ETF
            if CASH_ETF in closes_now.index:
                target[CASH_ETF] = self.exposure
        else:
            k = min(self.top_k, len(positive))
            ranked = sorted(positive, key=positive.__getitem__, reverse=True)[:k]

            # Check if breadth fails for equity ETFs
            equity_etfs = {"QQQ", "SPY", "IWM", "RSP", "VGT", "XLV"}
            if not breadth_ok:
                # Filter out equity ETFs, keep only TLT/GLD
                ranked = [s for s in ranked if s not in equity_etfs]
                if not ranked:
                    ranked = [CASH_ETF] if CASH_ETF in closes_now.index else []

            if not ranked:
                if CASH_ETF in closes_now.index:
                    target[CASH_ETF] = self.exposure
            else:
                per_weight = self.exposure / len(ranked)
                for sym in ranked:
                    target[sym] = per_weight

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


NAME = "etf_momentum_breadth"
HYPOTHESIS = (
    "ETF composite momentum with RSP/SPY breadth gate: rank QQQ/SPY/IWM/RSP/VGT/XLV/TLT/GLD "
    "by composite 1m+3m return, hold top-2 with positive absolute momentum; IEF if none qualify; "
    "breadth gate: skip equity ETFs when RSP lags SPY by >2% over 20d; monthly rebalance"
)
UNIVERSE = ["QQQ", "SPY", "IWM", "RSP", "VGT", "XLV", "TLT", "GLD", "IEF"]
STRATEGY = ETFMomentumBreadth()
