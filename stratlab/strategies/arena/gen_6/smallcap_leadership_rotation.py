"""Small-cap leadership rotation strategy.

Hypothesis: When small-cap stocks (IWM/Russell 2000) are outperforming
large-caps (SPY/S&P 500) on a rolling 20-day basis, it signals high
investor risk appetite and broad market participation — a historically
bullish environment. The opposite (large-cap leadership) signals defensiveness.

Signal: IWM 20d return vs SPY 20d return
  - IWM outperforming (risk-on): hold QQQ 60% + IWM 37%
    (tech growth + small-cap momentum = full risk-on)
  - SPY outperforming (risk-off): hold SPY 60% + TLT 37%
    (defensive rotation — large cap quality + bonds)
  - Weekly rebalance (every 5 bars)

Rationale: Small-cap stocks are more economically sensitive (higher beta,
more domestic revenue, more financial leverage). When they lead, it signals
genuine economic optimism and liquidity is flowing to riskier assets.
This is a breadth-quality signal distinct from:
  - VIX: measures fear/volatility, not relative leadership
  - Credit spreads: measures bond risk, not equity breadth
  - UUP (already on leaderboard from sonnet-9 submission): FX-based

Diversification vs leaderboard:
  - No prior strategy uses IWM/SPY relative return as regime signal
  - Two-state allocation (QQQ+IWM vs SPY+TLT) is different from all gen_5
  - IWM position provides small-cap exposure absent from leaderboard
  - Different corr path from dollar_strength_em_rotation (sonnet-9 accepted):
    this rotates within US equity size factors, not between US/EM
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

RS_WINDOW = 20     # 20d rolling return for relative strength comparison
REBALANCE = 5      # weekly
EXPOSURE = 0.97


class SmallCapLeadershipRotation(Strategy):
    """IWM vs SPY relative strength drives equity breadth rotation."""

    def __init__(
        self,
        rs_window: int = RS_WINDOW,
        rebalance: int = REBALANCE,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rs_window=rs_window,
            rebalance=rebalance,
            exposure=exposure,
        )
        self.rs_window = int(rs_window)
        self.rebalance = int(rebalance)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.rs_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance != 0:
            return []

        # Read IWM history for small-cap relative performance
        try:
            iwm_hist = ctx.history("IWM")
        except KeyError:
            return []
        if iwm_hist is None or len(iwm_hist) < self.rs_window + 2:
            return []
        iwm_close = iwm_hist["close"].dropna()
        if len(iwm_close) < self.rs_window + 1:
            return []

        # Read SPY history for large-cap reference
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if spy_hist is None or len(spy_hist) < self.rs_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.rs_window + 1:
            return []

        # Compute rolling returns
        iwm_ret = float(iwm_close.iloc[-1] / iwm_close.iloc[-self.rs_window - 1] - 1.0)
        spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.rs_window - 1] - 1.0)

        if not np.isfinite(iwm_ret) or not np.isfinite(spy_ret):
            return []

        # Regime: small caps outperforming = risk-on
        smallcap_leading = iwm_ret > spy_ret

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine allocation based on regime
        if smallcap_leading:
            # Risk-on: QQQ + IWM
            allocations = [("QQQ", 0.60 * self.exposure), ("IWM", 0.40 * self.exposure)]
        else:
            # Risk-off / large-cap defensive: SPY + TLT
            allocations = [("SPY", 0.60 * self.exposure), ("TLT", 0.40 * self.exposure)]

        target: dict[str, float] = {}
        for sym, weight in allocations:
            if sym in closes_now.index:
                target[sym] = weight

        # Fallback to SPY if target ETFs missing
        if not target:
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure

        # Build orders
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "smallcap_leadership_rotation"
HYPOTHESIS = (
    "Small-cap leadership rotation: use IWM vs SPY 20d relative strength MA as "
    "risk-appetite gauge; when IWM 20d return > SPY 20d return (small caps leading) "
    "hold QQQ 60%+IWM 37%; when large caps leading hold SPY 60%+TLT 37%; weekly "
    "rebalance; small-cap breadth as risk-on signal distinct from VIX/credit/yield signals"
)

UNIVERSE = ["IWM", "QQQ", "SPY", "TLT"]

STRATEGY = SmallCapLeadershipRotation()
