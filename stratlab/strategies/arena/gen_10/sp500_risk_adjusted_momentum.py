"""SP500 risk-adjusted momentum (information ratio proxy) — gen_10 sonnet-7

Hypothesis: Rank SP500 stocks by 63d return divided by 63d realized
volatility (an "information ratio" or risk-adjusted momentum score). This
selects stocks with the best risk-adjusted momentum — not just the fastest
movers, but the most efficient trending names. Stocks with high raw return
but high vol rank lower than stocks with moderate return but low vol.

Rationale:
  - Pure momentum: rank by 126d return. High vol names can dominate.
  - Risk-adjusted momentum: rank by 63d_return / 63d_vol. Selects stocks
    with the best "efficiency" of trending — similar to a Sharpe ratio over
    the momentum window.
  - This is different from both:
    - RSI quality filter (gen9): excludes by RSI<35 threshold
    - Low-vol ranking: selects by lowest vol regardless of momentum
    - The ratio captures both dimensions simultaneously.
  - 63d window (3 months) is shorter than gen9's 126d — different
    "speed" of momentum, so the candidate pool differs.
  - Per-stock 200d SMA trend gate to avoid selecting downtrending names
    that happen to have risen briefly on low vol.
  - SPY 200d outer bear gate to TLT.
  - Inverse-vol weighted for position sizing (separate from ranking).

Design:
  - Score = (63d_return) / (63d_realized_vol * sqrt(63)) — annualized IR.
  - Include only stocks above their own 200d SMA with positive 63d return.
  - Hold top-15 by score; inverse-vol (21d) weighted.
  - SPY 200d outer bear -> TLT.
  - Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
SCORE_WINDOW = 63         # risk-adjusted momentum window
TREND_WINDOW = 200        # per-stock SMA gate
VOL_WINDOW = 21           # inverse-vol for position sizing
SPY_TREND_WINDOW = 200    # outer bear gate
TOP_K = 15
EXPOSURE = 0.97


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "TLT"]


UNIVERSE = _universe


class SP500RiskAdjustedMomentum(Strategy):
    """SP500 stocks ranked by 63d return/63d_vol ratio (risk-adjusted momentum);
    positive-return and 200d SMA filters; inverse-vol weighted; SPY 200d outer
    bear gate to TLT; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        score_window: int = SCORE_WINDOW,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            score_window=score_window,
            trend_window=trend_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.score_window = int(score_window)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.score_window, self.trend_window, self.spy_trend_window) + 10
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
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            need = max(self.score_window, self.trend_window) + self.vol_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.score_window + 5:
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                scores: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in ("SPY", "TLT"):
                        continue
                    col = prices[sym].dropna()
                    n = len(col)
                    if n < max(self.score_window, self.trend_window) + self.vol_window + 2:
                        continue

                    # 63d momentum — must be positive
                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-self.score_window])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    mom_ret = p_end / p_start - 1.0
                    if not np.isfinite(mom_ret) or mom_ret <= 0:
                        continue

                    # Per-stock 200d SMA gate
                    sma_200 = float(col.iloc[-self.trend_window:].mean())
                    if p_end <= sma_200:
                        continue

                    # 63d realized vol for scoring
                    tail_score = col.values[-(self.score_window + 1):]
                    if len(tail_score) < self.score_window + 1:
                        continue
                    logr_score = np.log(tail_score[1:] / tail_score[:-1])
                    rv_score = float(np.std(logr_score))
                    if rv_score <= 1e-6 or not np.isfinite(rv_score):
                        continue

                    # Risk-adjusted momentum score = annualized return / annualized vol
                    ann_factor = np.sqrt(252.0 / self.score_window)
                    annual_ret = (1.0 + mom_ret) ** (252.0 / self.score_window) - 1.0
                    annual_vol = rv_score * np.sqrt(252.0)
                    ir_score = annual_ret / annual_vol
                    if not np.isfinite(ir_score):
                        continue

                    # 21d realized vol for position sizing
                    tail_pos = col.values[-(self.vol_window + 1):]
                    if len(tail_pos) < self.vol_window + 1:
                        continue
                    logr_pos = np.log(tail_pos[1:] / tail_pos[:-1])
                    rv_pos = float(np.std(logr_pos))
                    if rv_pos <= 1e-6 or not np.isfinite(rv_pos):
                        continue

                    scores[sym] = ir_score
                    inv_vols[sym] = 1.0 / rv_pos

                if len(scores) < 5:
                    if "TLT" in live:
                        target["TLT"] = self.exposure
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


NAME = "sp500_risk_adjusted_momentum"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock 21d return acceleration filter: exclude stocks where "
    "21d return < 0 but 126d return > 0 (momentum decelerating); hold top-15 remaining stocks "
    "above their 126d SMA; inverse-vol weighted; SPY 200d outer bear gate to TLT; biweekly "
    "rebalance — acceleration filter is orthogonal to RSI quality filter"
)

STRATEGY = SP500RiskAdjustedMomentum()
