"""Dual-bucket equity rotation: QQQ vs SPY based on NASDAQ leadership + VIX gate.

Hypothesis: Three-state regime driven by VIX and QQQ relative momentum:
  1. QQQ leadership (QQQ 20d return > SPY 20d return AND VIX below 20d MA):
     Hold QQQ 97% — growth/tech momentum environment
  2. Broad market (VIX below 20d MA but QQQ NOT leading):
     Hold SPY 97% — broad participation but tech not dominant
  3. Risk-off (VIX above 20d MA):
     Hold TLT 97% — elevated volatility signals defensive rotation

Rebalance weekly (every 5 bars) with a 3-bar minimum hold to avoid excessive
churn on VIX oscillations around the MA.

Rationale: The QQQ vs SPY 20-day relative return spread is a fast, direct
measure of growth/tech leadership. When tech leads AND volatility is calm,
QQQ provides higher returns than broad SPY. When vol is calm but tech is NOT
the leader, SPY participates in the breadth without concentration in tech. When
VIX spikes, the regime flips defensive to TLT.

Distinction from existing strategies:
  - jnk_vix_dual_gate_qqq uses credit (JNK) AND VIX, not QQQ leadership
  - qqq_vs_xlv_rotation uses 60d momentum between tech and healthcare
  - smallcap_leadership_rotation uses IWM vs SPY breadth, not QQQ vs SPY
  - tech_vs_defensive_rotation uses XLK vs XLU, not QQQ vs SPY with VIX overlay
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # bars (~1 week)
MIN_HOLD_BARS = 3       # minimum bars before rebalancing
RS_WINDOW = 20          # QQQ vs SPY relative strength window
VIX_MA_WINDOW = 20      # VIX moving average window for regime
EXPOSURE = 0.97
_VIX = "^VIX"


class QqqSpyVixLeadership(Strategy):
    """QQQ when tech leads + VIX calm; SPY when broad leads + VIX calm; TLT when VIX elevated."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        min_hold_bars: int = MIN_HOLD_BARS,
        rs_window: int = RS_WINDOW,
        vix_ma_window: int = VIX_MA_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            min_hold_bars=min_hold_bars,
            rs_window=rs_window,
            vix_ma_window=vix_ma_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.min_hold_bars = int(min_hold_bars)
        self.rs_window = int(rs_window)
        self.vix_ma_window = int(vix_ma_window)
        self.exposure = float(exposure)
        self._last_rebal = -999

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.rs_window, self.vix_ma_window) + 10
        if ctx.idx < warmup:
            return []

        # Enforce min hold
        bars_since_rebal = ctx.idx - self._last_rebal
        if bars_since_rebal < self.min_hold_bars:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # VIX regime check
        vix_calm = True
        try:
            vix_hist = ctx.history(_VIX)
            if vix_hist is not None and len(vix_hist) >= self.vix_ma_window + 5:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= self.vix_ma_window:
                    vix_current = float(vix_close.iloc[-1])
                    vix_ma = float(vix_close.iloc[-self.vix_ma_window:].mean())
                    vix_calm = vix_current < vix_ma
        except Exception:
            pass

        # QQQ vs SPY relative strength check
        qqq_leading = False
        if vix_calm:
            try:
                need = self.rs_window + 5
                prices = ctx.closes_window(need)
                if len(prices) >= self.rs_window and "QQQ" in prices.columns and "SPY" in prices.columns:
                    qqq_col = prices["QQQ"].dropna()
                    spy_col = prices["SPY"].dropna()
                    if len(qqq_col) >= self.rs_window and len(spy_col) >= self.rs_window:
                        qqq_ret = float(qqq_col.iloc[-1] / qqq_col.iloc[-self.rs_window] - 1.0)
                        spy_ret = float(spy_col.iloc[-1] / spy_col.iloc[-self.rs_window] - 1.0)
                        qqq_leading = np.isfinite(qqq_ret) and np.isfinite(spy_ret) and qqq_ret > spy_ret
            except Exception:
                pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine target allocation
        target: dict[str, float] = {}
        if not vix_calm:
            # Risk-off: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        elif qqq_leading:
            # Tech leadership + calm vol: QQQ
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        else:
            # Broad market (not tech-led) but calm: SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        self._last_rebal = ctx.idx

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


NAME = "qqq_spy_vix_leadership"
HYPOTHESIS = (
    "Dual-bucket equity rotation: QQQ when NASDAQ leadership (QQQ 20d return > SPY 20d return) "
    "AND VIX below 20d MA; SPY when broad market leads (VIX calm but QQQ not dominant); "
    "TLT 97% when VIX above 20d MA; weekly rebalance with min-3-bar hold"
)

UNIVERSE = ["QQQ", "SPY", "TLT", "SHY", _VIX]

STRATEGY = QqqSpyVixLeadership()
