"""^MOVE bond-vol 252d percentile gating SP500 cross-sectional momentum.

Hypothesis (sonnet-1, gen_10):
    ^MOVE bond-vol 252d percentile determines allocation:
    - Low MOVE pct (<40th): hold top-15 SP500 stocks by 126d-skip-21d momentum
      (inverse-vol weighted) — calm bond vol = good for risk assets
    - High MOVE pct (>70th): hold TLT 60% + IEF 37% — stressed bond vol = duration risk
    - Mid MOVE pct (40th-70th): SPY 97% — neutral
    SPY 200d outer bear gate to TLT. Biweekly rebalance.

Rationale:
  - gen9_opus5_move_bondvol_pct_gate (IS Calmar 0.58) proved ^MOVE percentile is
    a viable regime signal. That strategy allocated QQQ vs SPY; this one allocates
    to individual SP500 stocks vs bonds — a fundamentally different exposure.
  - Low MOVE → calm Treasuries → stable MBS and credit → risk-on environment that
    historically supports momentum stocks. The signal is regime-invariant: percentile
    rank self-calibrates to changing baseline MOVE levels.
  - High MOVE → bond vol stressed → duration risk high → TLT+IEF defensive blend
    rather than pure TLT (IEF provides shorter duration hedge).
  - ^MOVE vs ^VIX are structurally orthogonal: MOVE measures Treasury implied vol,
    VIX measures equity implied vol. Many periods have high MOVE + low VIX (e.g.
    2013 taper tantrum) or low MOVE + high VIX (equity-specific stress).
  - OOS expected retention: MODERATE-HIGH — MOVE percentile is structural and
    regime-invariant. However, the SP500 momentum component adds momentum risk.

Distinct from:
  - gen9_opus5_move_bondvol_pct_gate: uses QQQ/SPY ETFs, not stock selection
  - All gen7/8/9 gated momentum strategies: use VIX/credit/yield signals as gate
  - gen9_sp500_rsi_quality_momentum: uses RSI quality filter (per-stock), not regime gate
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOVE_WINDOW = 252       # 1-year rolling window for MOVE percentile
MOVE_LOW_PCT = 0.40     # below 40th pct = calm bond vol → stocks
MOVE_HIGH_PCT = 0.70    # above 70th pct = stressed bond vol → bonds
MOM_LOOKBACK = 126      # 6-month momentum
MOM_SKIP = 21           # skip last 1 month (Jegadeesh-Titman)
VOL_WINDOW = 21         # for inverse-vol weights
SPY_TREND_WINDOW = 200  # outer bear gate
TOP_K = 15
EXPOSURE = 0.97

# Defensive allocations
W_TLT_HIGH = 0.60
W_IEF_HIGH = 0.37


class MovePctSP500Momentum(Strategy):
    """^MOVE percentile gates: low MOVE → SP500 momentum, high MOVE → bonds, mid → SPY.

    SPY 200d outer bear gate to TLT. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        move_window: int = MOVE_WINDOW,
        move_low_pct: float = MOVE_LOW_PCT,
        move_high_pct: float = MOVE_HIGH_PCT,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            move_window=move_window,
            move_low_pct=move_low_pct,
            move_high_pct=move_high_pct,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.move_window = int(move_window)
        self.move_low_pct = float(move_low_pct)
        self.move_high_pct = float(move_high_pct)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.move_window, self.mom_lookback + self.mom_skip) + self.vol_window + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
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
            # Outer bear gate: full TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # --- Compute ^MOVE 252d percentile ---
            move_pct_rank = 0.5  # default to mid if not available
            try:
                move_hist = ctx.history("^MOVE")
                move_close = move_hist["close"].dropna()
                if len(move_close) >= self.move_window + 2:
                    current_move = float(move_close.iloc[-1])
                    window_vals = move_close.iloc[-self.move_window:].values
                    move_pct_rank = float(np.mean(window_vals <= current_move))
            except KeyError:
                pass

            if move_pct_rank > self.move_high_pct:
                # High bond vol: defensive bond blend
                if "TLT" in closes_now.index:
                    target["TLT"] = W_TLT_HIGH
                if "IEF" in closes_now.index:
                    target["IEF"] = W_IEF_HIGH

            elif move_pct_rank < self.move_low_pct:
                # Low bond vol: SP500 momentum stock selection
                need = self.mom_lookback + self.mom_skip + self.vol_window + 2
                prices = ctx.closes_window(need)
                if len(prices) < need - 2:
                    # Fallback to SPY
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    inv_vols: dict[str, float] = {}

                    for sym in prices.columns:
                        col = prices[sym].dropna()
                        if len(col) < self.mom_lookback + self.mom_skip:
                            continue
                        # Skip-month momentum
                        p_end = float(col.iloc[-self.mom_skip - 1])
                        p_start = float(col.iloc[-(self.mom_lookback + self.mom_skip)])
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
                # Mid-range MOVE: neutral SPY
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure

        # --- Build orders ---
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
    return sp500_tickers() + ["SPY", "TLT", "IEF", "^MOVE"]


NAME = "move_pct_sp500_momentum"
HYPOTHESIS = (
    "^MOVE 252d percentile gates SP500 momentum: low MOVE pct (<40th) hold top-15 SP500 by "
    "126d-skip-21d momentum (inverse-vol weighted); high MOVE pct (>70th) hold TLT 60%+IEF 37%; "
    "mid-MOVE hold SPY 97%; SPY 200d outer bear gate to TLT; biweekly rebalance — "
    "MOVE-percentile-gated stock selection orthogonal to all equity-vol (VIX) gated strategies"
)

UNIVERSE = _universe

STRATEGY = MovePctSP500Momentum()
