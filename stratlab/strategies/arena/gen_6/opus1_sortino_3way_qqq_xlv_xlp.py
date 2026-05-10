"""opus-1 mutation of qqq_vs_xlv_rotation (sector-binary cluster).

Parent: gen6_qqq_vs_xlv_rotation (IS Calmar 0.67, h2≈h1, corr_to_top5 0.61).

Structural mutations vs parent:
  - Universe: 2-asset binary (QQQ vs XLV)  ->  3-asset rotation
              (QQQ growth, XLV defensive-growth, XLP staples). 3rd defensive
              sector adds a second non-tech regime to switch into.
  - Signal:   60d total return  ->  90d Sharpe ratio (Sharpe-rank, not
              total-return-rank). Sharpe-ranking penalizes high-vol leaders;
              ranking under volatility is structurally different from
              ranking under return alone.
  - Bear gate: none (parent always in equities)  ->  if leader 90d return
              negative, hold IEF (one bond escape valve, not parent's all-
              equity rule).
  - Min hold:  5  ->  10 bars (slower 90d window justifies less churn).
  - Sizing:    100% leader  ->  100% leader at 0.97 exposure (single asset
              concentration retained — keeps the parent's binary character
              but with a different ranking signal & defensive escape).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

CANDIDATES = ["QQQ", "XLV", "XLP"]
SHARPE_WINDOW = 90
MIN_HOLD_BARS = 10
EXPOSURE = 0.97


class SharpeRank3WayRotation(Strategy):
    def __init__(
        self,
        sharpe_window: int = SHARPE_WINDOW,
        min_hold_bars: int = MIN_HOLD_BARS,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            sharpe_window=sharpe_window,
            min_hold_bars=min_hold_bars,
            exposure=exposure,
        )
        self.sharpe_window = int(sharpe_window)
        self.min_hold_bars = int(min_hold_bars)
        self.exposure = float(exposure)
        self._current_holding: str | None = None
        self._bars_since_switch: int = 0

    def _sharpe(self, close: np.ndarray) -> tuple[float, float]:
        """Annualized Sharpe and total return over the window."""
        if len(close) < 5:
            return 0.0, 0.0
        rets = np.diff(np.log(close))
        rets = rets[np.isfinite(rets)]
        if len(rets) < 5:
            return 0.0, 0.0
        sigma = float(np.std(rets))
        if sigma <= 1e-9:
            return 0.0, 0.0
        mean_r = float(np.mean(rets))
        sharpe = (mean_r * 252.0) / (sigma * np.sqrt(252.0))
        # Cumulative log-return over window
        total_ret = float(close[-1] / close[0] - 1.0) if close[0] > 0 else 0.0
        return sharpe, total_ret

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sharpe_window + 5
        if ctx.idx < warmup:
            return []

        sharpes: dict[str, float] = {}
        rets90: dict[str, float] = {}
        for sym in CANDIDATES:
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            if hist is None or len(hist) < self.sharpe_window + 1:
                continue
            close = hist["close"].dropna().values[-(self.sharpe_window + 1):]
            if len(close) < self.sharpe_window + 1:
                continue
            s, r = self._sharpe(close)
            sharpes[sym] = s
            rets90[sym] = r

        if len(sharpes) < 2:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Rank by Sharpe descending
        ranked = sorted(sharpes, key=sharpes.__getitem__, reverse=True)
        leader = ranked[0]
        leader_ret = rets90[leader]

        self._bars_since_switch += 1

        # Defensive escape: if leader 90d total return negative, go to IEF
        if leader_ret < 0.0:
            target_sym = "IEF" if "IEF" in live else None
        else:
            target_sym = leader

        # Min-hold guard
        if (
            self._current_holding is not None
            and target_sym != self._current_holding
            and self._bars_since_switch < self.min_hold_bars
        ):
            return []

        if target_sym is None or target_sym not in live:
            return []

        target = {target_sym: self.exposure}

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        for sym, weight in target.items():
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        if orders and target_sym != self._current_holding:
            self._current_holding = target_sym
            self._bars_since_switch = 0

        return orders


NAME = "opus1_sortino_3way_qqq_xlv_xlp"
HYPOTHESIS = (
    "Mutate qqq_vs_xlv_rotation: 3-way QQQ/XLV/XLP rotation by 90d Sharpe-rank "
    "(volatility-normalized) replaces binary 60d total-return; IEF escape when "
    "leader 90d return negative; 10-bar min hold; full leader exposure 0.97."
)
UNIVERSE = ["QQQ", "XLV", "XLP", "IEF"]

STRATEGY = SharpeRank3WayRotation()
