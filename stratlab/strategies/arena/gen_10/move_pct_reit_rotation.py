"""^MOVE bond-vol 252d percentile gating REIT subsegment rotation.

Hypothesis (sonnet-1, gen_10):
    ^MOVE bond-vol 252d percentile gates between REIT subsegments and equity:
    - Low MOVE pct (<33rd): REM 60% + SPY 37% (mortgage REITs benefit from
      calm bond vol — lower MBS yield volatility = higher mortgage spreads)
    - High MOVE pct (>67th): VNQ 60% + IEF 37% (equity REITs + rate buffer)
    - Mid MOVE pct (33rd-67th): SPY 97% (neutral equity)
    SPY 200d SMA outer bear gate to TLT. Weekly rebalance.

Rationale:
  - ^MOVE is the ICE BofAML bond volatility index — measures implied vol of
    US Treasuries. Orthogonal to ^VIX (equity vol). gen9_opus5_move_bondvol_pct_gate
    (IS Calmar 0.58) used MOVE percentile as a QQQ-or-SPY allocator.
    This extends to REIT subsegmentation — a different exposure angle.
  - REM (mortgage REITs) has different sensitivity to bond vol than VNQ
    (equity REITs): when MOVE is low (Treasury vol calm), MBS markets are
    stable, mortgage rates predictable → REM outperforms. When MOVE is high,
    equity REITs (VNQ) benefit from falling duration + IEF provides rate buffer.
  - REM and VNQ not used as tradeable allocators in any leaderboard strategy.
  - ^MOVE percentile is regime-invariant: it compares current MOVE to its
    own 252d history, so it adapts to changing baseline bond-vol levels.
    OOS retention expected: MODERATE-HIGH — MOVE is a genuine structural signal
    (not just a calm-IS regime artifact).

Data coverage:
  - REM: covers IS (from 2007)
  - VNQ: covers IS (from 2004)
  - ^MOVE: signal-only (non-tradeable), accessible via ctx.history("^MOVE")

Distinct from:
  - gen9_opus5_move_bondvol_pct_gate: allocates QQQ vs SPY (equity ETFs);
    this allocates REIT subsegments — entirely different exposure.
  - All gen7/8/9 REIT strategies: none use REM/VNQ as explicit allocation targets.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
MOVE_WINDOW = 252       # 1-year rolling window for MOVE percentile
MOVE_LOW_PCT = 0.33     # below 33rd pct = calm bond vol
MOVE_HIGH_PCT = 0.67    # above 67th pct = stressed bond vol
SPY_TREND_WINDOW = 200  # outer bear gate
EXPOSURE = 0.97

# Allocation weights for each regime
# Low MOVE: mortgage REITs (calm) + broad equity
W_REM_LOW = 0.60
W_SPY_LOW = 0.37
# High MOVE: equity REITs + rate buffer
W_VNQ_HIGH = 0.60
W_IEF_HIGH = 0.37
# Mid MOVE: pure equity
W_SPY_MID = 0.97


class MovePctReitRotation(Strategy):
    """^MOVE bond-vol percentile gating between REIT subsegments and equity.

    Low MOVE (<33rd pct): REM 60% + SPY 37%
    High MOVE (>67th pct): VNQ 60% + IEF 37%
    Mid MOVE: SPY 97%
    SPY 200d bear gate to TLT. Weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        move_window: int = MOVE_WINDOW,
        move_low_pct: float = MOVE_LOW_PCT,
        move_high_pct: float = MOVE_HIGH_PCT,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            move_window=move_window,
            move_low_pct=move_low_pct,
            move_high_pct=move_high_pct,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.move_window = int(move_window)
        self.move_low_pct = float(move_low_pct)
        self.move_high_pct = float(move_high_pct)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.move_window + 10
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
            # Bear: TLT full defensive
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # --- Compute ^MOVE 252d percentile ---
            try:
                move_hist = ctx.history("^MOVE")
            except KeyError:
                # If ^MOVE not available, fall back to SPY allocation
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
                target = {"SPY": self.exposure}
            else:
                move_close = move_hist["close"].dropna()
                if len(move_close) < self.move_window + 2:
                    # Not enough MOVE history — use neutral SPY
                    target = {"SPY": W_SPY_MID}
                else:
                    current_move = float(move_close.iloc[-1])
                    window_vals = move_close.iloc[-self.move_window:].values
                    # Percentile rank of current MOVE within rolling window
                    pct_rank = float(np.mean(window_vals <= current_move))

                    if pct_rank < self.move_low_pct:
                        # Low bond vol: mortgage REITs + SPY
                        if "REM" in closes_now.index and "SPY" in closes_now.index:
                            target["REM"] = W_REM_LOW
                            target["SPY"] = W_SPY_LOW
                        elif "VNQ" in closes_now.index:
                            target["VNQ"] = self.exposure
                        else:
                            target["SPY"] = self.exposure

                    elif pct_rank > self.move_high_pct:
                        # High bond vol: equity REITs + rate buffer
                        if "VNQ" in closes_now.index and "IEF" in closes_now.index:
                            target["VNQ"] = W_VNQ_HIGH
                            target["IEF"] = W_IEF_HIGH
                        elif "VNQ" in closes_now.index:
                            target["VNQ"] = self.exposure
                        else:
                            target["IEF"] = self.exposure

                    else:
                        # Mid: neutral equity
                        if "SPY" in closes_now.index:
                            target["SPY"] = W_SPY_MID

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


UNIVERSE = ["REM", "VNQ", "SPY", "TLT", "IEF", "^MOVE"]

NAME = "move_pct_reit_rotation"
HYPOTHESIS = (
    "^MOVE bond-vol 252d percentile gates REIT subsegment rotation: "
    "low MOVE pct (<33rd) hold REM 60%+SPY 37% (mortgage REITs benefit from calm rates), "
    "high MOVE pct (>67th) hold VNQ 60%+IEF 37% (equity REITs + rate buffer), "
    "mid-MOVE hold SPY 97%; SPY 200d bear gate to TLT; weekly rebalance — "
    "combines successful gen9 MOVE-percentile angle with unexplored REM/VNQ subsegment"
)

STRATEGY = MovePctReitRotation()
