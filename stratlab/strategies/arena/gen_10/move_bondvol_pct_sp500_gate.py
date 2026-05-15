"""^MOVE bond-vol 252d percentile gate on SP500 momentum — gen_10 sonnet-7

Hypothesis: Use the ^MOVE index (ICE BofA MOVE index — bond market implied
volatility, the "VIX for bonds") 252-day percentile as a macro regime gate.
When bond volatility is in low-percentile territory (<50th pctile), rate
uncertainty is calm — hold top-15 SP500 stocks by 63d momentum. When bond
vol is elevated (>75th pctile), rate uncertainty is high — hold TLT 97%.
Middle regime (50-75th pctile): hold SPY 60%+TLT 37%. SPY 200d outer bear
gate always goes to TLT. Inverse-vol weighted for stock positions.

Rationale:
  - ^MOVE captures bond-market fear distinct from equity-market fear (^VIX).
    High MOVE = rising rate uncertainty, which hurts equity multiples
    disproportionately. Low MOVE = calm rates, equity expansion works.
  - gen9_opus5_move_bondvol_pct_gate used ^MOVE percentile on QQQ/SPY
    allocator (IS Calmar 0.58). This uses MOVE percentile on SP500 STOCK
    SELECTION (not ETF switching), which has a different return distribution
    and higher potential IS Calmar.
  - Percentile form (not absolute MOVE level) avoids regime sensitivity to
    whether MOVE is "high" in absolute terms — relative to its own 252d
    history is the right comparison.
  - Three-tier regime (calm/middle/elevated) vs gen9's binary provides more
    nuanced exposure management.
  - ^MOVE covers IS window (starts 2002). All other ETFs cover IS.

Design:
  - MOVE 252d percentile computed on each bar from rolling 252d window.
  - Percentile < 0.50: calm bond vol -> SP500 top-15 63d momentum.
  - Percentile 0.50-0.75: middle -> SPY 60%+TLT 37%.
  - Percentile > 0.75: elevated bond vol -> TLT 97%.
  - SPY 200d outer bear: override to TLT 97%.
  - Rebalance every 10 bars (biweekly).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOVE_PCT_WINDOW = 252      # 252-day percentile window
CALM_THRESHOLD = 0.50      # percentile below -> calm (stock momentum)
ELEVATED_THRESHOLD = 0.75  # percentile above -> elevated (TLT)
MOMENTUM_WINDOW = 63       # stock momentum lookback
VOL_WINDOW = 21            # inverse-vol for stock sizing
SPY_TREND_WINDOW = 200     # outer bear gate
TOP_K = 15
EXPOSURE = 0.97
NEUTRAL_SPY_W = 0.60       # SPY weight in middle regime
NEUTRAL_TLT_W = 0.37       # TLT weight in middle regime


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["^MOVE", "SPY", "TLT"]


UNIVERSE = _universe


class MoveBondvolPctSP500Gate(Strategy):
    """^MOVE 252d percentile three-tier gate: calm -> SP500 momentum;
    middle -> SPY+TLT blend; elevated -> TLT; SPY 200d outer bear gate;
    inverse-vol weighted; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        move_pct_window: int = MOVE_PCT_WINDOW,
        calm_threshold: float = CALM_THRESHOLD,
        elevated_threshold: float = ELEVATED_THRESHOLD,
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
            move_pct_window=move_pct_window,
            calm_threshold=calm_threshold,
            elevated_threshold=elevated_threshold,
            momentum_window=momentum_window,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            neutral_spy_w=neutral_spy_w,
            neutral_tlt_w=neutral_tlt_w,
        )
        self.rebalance_every = int(rebalance_every)
        self.move_pct_window = int(move_pct_window)
        self.calm_threshold = float(calm_threshold)
        self.elevated_threshold = float(elevated_threshold)
        self.momentum_window = int(momentum_window)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.neutral_spy_w = float(neutral_spy_w)
        self.neutral_tlt_w = float(neutral_tlt_w)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.spy_trend_window, self.move_pct_window) + 10
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
            # Compute ^MOVE 252d percentile
            move_pct = 0.5  # default: middle regime if no data
            try:
                move_hist = ctx.history("^MOVE")
                if len(move_hist) >= self.move_pct_window + 2:
                    move_close = move_hist["close"].dropna()
                    if len(move_close) >= self.move_pct_window + 1:
                        window = move_close.values[-self.move_pct_window:]
                        current_val = float(move_close.iloc[-1])
                        n_below = float(np.sum(window < current_val))
                        move_pct = n_below / len(window)
            except (KeyError, Exception):
                pass

            if move_pct > self.elevated_threshold:
                # Elevated bond vol: TLT defensive
                if "TLT" in live:
                    target["TLT"] = self.exposure
            elif move_pct > self.calm_threshold:
                # Middle regime: SPY + TLT blend
                if "SPY" in live:
                    target["SPY"] = self.neutral_spy_w
                if "TLT" in live:
                    target["TLT"] = self.neutral_tlt_w
            else:
                # Calm bond vol: SP500 stock momentum
                need = self.momentum_window + self.vol_window + 5
                prices = ctx.closes_window(need)
                if len(prices) < self.momentum_window + 5:
                    if "TLT" in live:
                        target["TLT"] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}

                    for sym in prices.columns:
                        if sym in ("SPY", "TLT", "^MOVE"):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.momentum_window + self.vol_window + 2:
                            continue

                        # 63d momentum
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


NAME = "move_bondvol_pct_sp500_gate"
HYPOTHESIS = (
    "^MOVE bond-vol 252d percentile regime gate with SP500 stock selection: when MOVE "
    "percentile < 0.5 (calm bond vol, rate uncertainty low), hold top-15 SP500 stocks by "
    "63d momentum above 200d SMA; when MOVE percentile > 0.75 (elevated bond vol, rates "
    "stressed), hold TLT 97pct; middle regime (0.5-0.75) hold SPY 60pct+TLT 37pct; "
    "inverse-vol weighted; biweekly rebalance"
)

STRATEGY = MoveBondvolPctSP500Gate()
