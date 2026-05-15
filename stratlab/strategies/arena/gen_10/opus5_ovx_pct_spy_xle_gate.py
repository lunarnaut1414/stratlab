"""^OVX (oil VIX) 252d percentile gates SPY+XLE risk-on vs TLT+AGG defensive.

Hypothesis (opus-5, gen_10 wildcard, anti-consensus):
    ^OVX (CBOE Crude Oil Volatility Index) 252d rolling percentile determines
    allocation between two passive-index sleeves:
      - Low OVX pct (<30th):  SPY 50% + XLE 47%   (calm oil → energy sector
                                                    risk-on, broad equity)
      - High OVX pct (>70th): TLT 60% + AGG 37%   (oil shock → duration safe
                                                    haven + broad-bond mix)
      - Mid OVX pct (30-70):  SPY 95%             (neutral equity)
    SPY 200d outer bear gate to TLT. Biweekly rebalance.

Anti-consensus rationale:
  - ^OVX is the implied volatility of crude oil prices — a regime signal that
    is STRUCTURALLY ORTHOGONAL to all three vol indices used in the leaderboard:
    * VIX  (equity implied vol — saturated since gen_5)
    * MOVE (Treasury implied vol — mainstream since gen_9, 2 variants in gen_10)
    * SKEW (equity tail risk — failed in gen_9, all tunings rejected)
    OVX captures commodity/geopolitical/inflation regime that the other three
    miss. Recent examples: 2014-16 oil crash (OVX spiked while VIX stayed muted),
    2018 Q4 oil break (OVX moved before VIX).
  - The phase2_brief explicitly lists ^OVX 252d percentile as "GENUINELY
    UNTOUCHED" after 10 generations of arena evolution.
  - Routes to SPY+XLE risk-on (NOT SP500 cross-sectional momentum — the dominant
    saturated cluster with 15 variants in gen_10 alone) and TLT+AGG defensive
    (NOT pure TLT or TLT+IEF — different blend from gen_10 strategies).
  - Calm oil-vol (<30th pct) historically correlates with risk-appetite recovery
    AFTER stress events, and energy-sector outperformance during demand-led
    growth regimes. High OVX (>70th pct) signals oil-shock or geopolitical
    stress where duration outperforms equity.
  - Percentile-rank self-calibrates to changing baseline OVX levels (e.g.
    2017's compressed 25-40 range vs 2015-16's elevated 40-75 range).

Distinct from:
  - gen10_move_pct_sp500_momentum (IS 0.61): MOVE pct → SP500 stock selection
    (we use SPY+XLE pair, no stock selection)
  - gen10_move_bondvol_pct_sp500_gate (IS 0.66): MOVE pct → QQQ/SPY rotation
    (we use OVX, different signal axis, different ETFs)
  - gen10_gdx_gld_sector_timing: gold-vs-miners → XLK+XLE sectors
    (we gate on oil-vol, not gold-miner ratio; SPY broad equity not XLK)
  - All VIX-gated strategies: equity-vol regime, not oil-vol regime
  - All SP500 cross-sectional momentum variants: no per-stock selection here

Hard constraints honored: allow_short=False, enforce_cash=True, IS only.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
OVX_WINDOW = 252            # 1-year rolling window for OVX percentile
OVX_LOW_PCT = 0.30          # below 30th pct = calm oil-vol → risk-on
OVX_HIGH_PCT = 0.70         # above 70th pct = stressed oil-vol → defensive
SPY_TREND_WINDOW = 200      # outer bear gate
EXPOSURE = 0.97

# Risk-on sleeve (low OVX pct): SPY + XLE
W_SPY_LOW = 0.50
W_XLE_LOW = 0.47

# Defensive sleeve (high OVX pct): TLT + AGG blend (not pure TLT — distinct)
W_TLT_HIGH = 0.60
W_AGG_HIGH = 0.37


class OVXPctSPYXLEGate(Strategy):
    """^OVX 252d percentile gates passive-index sleeves.

    Low OVX pct: SPY+XLE risk-on. High OVX pct: TLT+AGG defensive.
    Mid: SPY neutral. SPY 200d outer bear gate to TLT. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        ovx_window: int = OVX_WINDOW,
        ovx_low_pct: float = OVX_LOW_PCT,
        ovx_high_pct: float = OVX_HIGH_PCT,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            ovx_window=ovx_window,
            ovx_low_pct=ovx_low_pct,
            ovx_high_pct=ovx_high_pct,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.ovx_window = int(ovx_window)
        self.ovx_low_pct = float(ovx_low_pct)
        self.ovx_high_pct = float(ovx_high_pct)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.ovx_window, self.spy_trend_window) + 10
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
            # Outer bear gate: full TLT (duration hedge)
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # --- Compute ^OVX 252d percentile ---
            ovx_pct_rank: float | None = None
            try:
                ovx_hist = ctx.history("^OVX")
                ovx_close = ovx_hist["close"].dropna()
                if len(ovx_close) >= self.ovx_window + 2:
                    current_ovx = float(ovx_close.iloc[-1])
                    window_vals = ovx_close.iloc[-self.ovx_window:].values
                    ovx_pct_rank = float(np.mean(window_vals <= current_ovx))
            except KeyError:
                ovx_pct_rank = None

            if ovx_pct_rank is None:
                # OVX not available — neutral SPY fallback
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            elif ovx_pct_rank > self.ovx_high_pct:
                # High oil-vol: defensive TLT + AGG blend
                if "TLT" in closes_now.index:
                    target["TLT"] = W_TLT_HIGH
                if "AGG" in closes_now.index:
                    target["AGG"] = W_AGG_HIGH
            elif ovx_pct_rank < self.ovx_low_pct:
                # Low oil-vol: SPY + XLE risk-on
                if "SPY" in closes_now.index:
                    target["SPY"] = W_SPY_LOW
                if "XLE" in closes_now.index:
                    target["XLE"] = W_XLE_LOW
            else:
                # Mid-range OVX: neutral SPY
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure

        # --- Build orders ---
        orders: list[Order] = []

        # Close any positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Open / adjust target positions
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
    # Tight universe: only the ETFs we trade + ^OVX signal
    return ["SPY", "XLE", "TLT", "AGG", "^OVX"]


NAME = "opus5_ovx_pct_spy_xle_gate"
HYPOTHESIS = (
    "^OVX 252d percentile gates passive sleeves: low OVX pct (<30th) SPY 50%+XLE 47% "
    "(calm oil → energy risk-on); high OVX pct (>70th) TLT 60%+AGG 37% (oil shock → "
    "duration+broad bonds); mid SPY 95%; SPY 200d outer bear gate to TLT; biweekly "
    "rebalance — oil-vol regime orthogonal to VIX/MOVE/SKEW, anti-consensus angle"
)

UNIVERSE = _universe

STRATEGY = OVXPctSPYXLEGate()
