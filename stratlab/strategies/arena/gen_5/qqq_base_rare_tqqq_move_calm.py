"""QQQ-base with rare-event TQQQ replacement during ^MOVE-z extreme calm.

Hypothesis (Phase-2 wildcard, third attempt):
  Two prior wildcard attempts (5-signal composite, dynamic SH overlay)
  fell short of 0.5 IS Calmar — the issue was either too-frequent regime
  switching or the hedge added drag during a 9yr bull. Lesson: in 2010-18,
  the right shape is "stay long tech, only de-risk on actual macro stress,
  and only USE leverage during rare clean-calm windows".

  This strategy:
    - Default: QQQ 97% (always-invested tech long, captures the bull)
    - Stress: SPY 200d SMA breach → TLT 60% + SHY 37% (off-equity)
    - Rare-event TQQQ replacement: when ^MOVE 60d z-score is *below* -0.7
      (extreme bond-vol calm — typically <10% of bars), substitute the QQQ
      sleeve with TQQQ 50% + QQQ 47%. The replacement holds for at least
      10 bars to avoid whipsaw.
    - Rebalance every 5 bars

  This is structurally different from opus-2's "TQQQ-trend-VIX-MOVE" intent
  because:
    - QQQ is the BASE state, not a binary regime; TQQQ is a *rare overlay*
    - Trigger is a SIGNED z-score (rolling 60d) on ^MOVE, not a hard level
    - Lockstep window prevents brake/flicker
    - SPY 200d gate is the only stress gate; no VIX threshold

Why anti-consensus / wildcard:
  - QQQ-default-with-rare-leveraged-overlay is structurally distinct from
    every leaderboard pattern (which all rotate or gate, never overlay)
  - Rare-event TQQQ trigger (~10% of bars) limits leverage exposure to
    the cleanest calm windows — exactly when 3x leverage compounds best
  - Uses ^MOVE z-score rather than ^MOVE absolute level
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

_MOVE = "^MOVE"
_TQQQ = "TQQQ"
_QQQ = "QQQ"
_SPY = "SPY"
_TLT = "TLT"
_SHY = "SHY"

Z_LOOKBACK = 60
TREND_WINDOW = 200
REBALANCE_EVERY = 5
EXPOSURE = 0.97
MOVE_Z_TRIGGER = -0.7    # extreme bond-vol calm
LEVERAGE_LOCKSTEP = 10   # bars to hold once triggered


def _zscore_last(series: pd.Series, lookback: int) -> float:
    if series is None or len(series) < lookback + 1:
        return float("nan")
    window = series.iloc[-lookback:]
    mu = float(window.mean())
    sigma = float(window.std(ddof=0))
    if sigma <= 0 or not np.isfinite(sigma):
        return float("nan")
    return (float(series.iloc[-1]) - mu) / sigma


class QqqBaseRareTqqqMoveCalm(Strategy):
    """QQQ default → TQQQ rare-event replacement on ^MOVE z<-0.7 → TLT/SHY stress."""

    def __init__(
        self,
        z_lookback: int = Z_LOOKBACK,
        trend_window: int = TREND_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
        move_z_trigger: float = MOVE_Z_TRIGGER,
        leverage_lockstep: int = LEVERAGE_LOCKSTEP,
    ) -> None:
        super().__init__(
            z_lookback=z_lookback,
            trend_window=trend_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
            move_z_trigger=move_z_trigger,
            leverage_lockstep=leverage_lockstep,
        )
        self.z_lookback = z_lookback
        self.trend_window = trend_window
        self.rebalance_every = rebalance_every
        self.exposure = exposure
        self.move_z_trigger = move_z_trigger
        self.leverage_lockstep = leverage_lockstep
        self._leverage_until_idx = -1

    def _series(self, ctx: BarContext, symbol: str) -> pd.Series | None:
        try:
            hist = ctx.history(symbol)
        except Exception:
            return None
        if hist is None or len(hist) == 0 or "close" not in hist.columns:
            return None
        s = hist["close"].dropna()
        return s if len(s) > 0 else None

    def _spy_above_trend(self, ctx: BarContext) -> bool:
        s = self._series(ctx, _SPY)
        if s is None or len(s) < self.trend_window + 1:
            return True   # default to trend-up if no signal yet
        sma = float(s.iloc[-self.trend_window:].mean())
        last = float(s.iloc[-1])
        return last >= sma

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.z_lookback, self.trend_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live_closes = {s: float(p) for s, p in closes_now.items()}
        portfolio_value = ctx.portfolio_value(live_closes)
        if portfolio_value <= 0:
            return []

        target: dict[str, float] = {}

        if not self._spy_above_trend(ctx):
            # Stress: SPY below 200d SMA → bonds + cash
            if _TLT in closes_now.index and _SHY in closes_now.index:
                target[_TLT] = 0.60 * self.exposure
                target[_SHY] = 0.37 * self.exposure
            elif _SHY in closes_now.index:
                target[_SHY] = self.exposure
        else:
            # Bull regime: QQQ default; check rare-event TQQQ trigger
            move_s = self._series(ctx, _MOVE)
            move_z = (
                _zscore_last(move_s, self.z_lookback)
                if move_s is not None
                else float("nan")
            )
            engage_leverage = (
                np.isfinite(move_z)
                and move_z <= self.move_z_trigger
                and _TQQQ in closes_now.index
                and _QQQ in closes_now.index
            )
            if engage_leverage:
                self._leverage_until_idx = ctx.idx + self.leverage_lockstep
            in_lockstep = (
                ctx.idx < self._leverage_until_idx
                and _TQQQ in closes_now.index
                and _QQQ in closes_now.index
            )
            if engage_leverage or in_lockstep:
                target[_TQQQ] = 0.50 * self.exposure
                target[_QQQ] = 0.47 * self.exposure
            elif _QQQ in closes_now.index:
                target[_QQQ] = self.exposure

        if not target:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live_closes.get(sym)
            if not price or price <= 0:
                continue
            target_shares = int(portfolio_value * weight / price)
            current_pos = int(ctx.position(sym).size)
            delta = target_shares - current_pos
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


def _universe() -> list[str]:
    return [_TQQQ, _QQQ, _SPY, _TLT, _SHY, _MOVE]


NAME = "qqq_base_rare_tqqq_move_calm"
HYPOTHESIS = (
    "QQQ-base with rare-event TQQQ replacement when ^MOVE 60d z<-0.7: hold QQQ 97% "
    "by default; when ^MOVE 60d z-score below -0.7 (extreme bond-vol calm) replace "
    "QQQ with TQQQ 50%+QQQ 47% for 10-bar lockstep window; TLT 60%+SHY 37% when "
    "SPY below 200d SMA. Rare-event leverage tilt untouched."
)
UNIVERSE = _universe

STRATEGY = QqqBaseRareTqqqMoveCalm()
