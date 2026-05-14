"""RSP/SPY breadth-based sector tilt strategy.

Hypothesis: Equal-weight RSP outperforming cap-weight SPY signals broad market
participation (risk-on). In that regime, rotate into top-2 cyclical sector ETFs
by 42d momentum. When large-caps lead (narrow leadership), hold SPY+TLT
defensively.

Rationale: RSP/SPY relative strength is an internal market breadth signal —
when smaller/more equal weighted stocks lead, it shows broad participation in
the rally. Narrow leadership (mega-caps dominating) often precedes corrections.
This is orthogonal to VIX level, credit spread signals, and gold regime signals.

Distinction from existing strategies:
  - RSP/SPY breadth ratio as primary regime signal (gen5_atr_momentum_etf
    uses RSP/SPY but routes to QQQ/SPY/TLT, not sector ETFs)
  - This routes breadth signal into sector rotation among XLK/XLY/XLI/XLF/XLB
  - Weekly rebalance with 30d breadth window
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5     # weekly
BREADTH_WINDOW = 30     # 30d for RSP vs SPY comparison
SECTOR_MOM_WINDOW = 42  # 42d sector momentum ranking
TOP_K = 2               # top-2 sectors
EXPOSURE = 0.97

SECTOR_ETFS = ["XLK", "XLY", "XLI", "XLF", "XLB", "XLE", "XLV", "XLU", "XLP"]


class RspBreadthSectorTilt(Strategy):
    """RSP/SPY breadth regime -> sector ETF rotation when broad market leads, else SPY+TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        breadth_window: int = BREADTH_WINDOW,
        sector_mom_window: int = SECTOR_MOM_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            breadth_window=breadth_window,
            sector_mom_window=sector_mom_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.breadth_window = int(breadth_window)
        self.sector_mom_window = int(sector_mom_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sector_mom_window + self.breadth_window + 10
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

        # Compute RSP vs SPY relative return over breadth_window
        broad_participation = True  # default
        try:
            rsp_hist = ctx.history("RSP")
            spy_hist = ctx.history("SPY")
            if (rsp_hist is not None and spy_hist is not None
                    and len(rsp_hist) >= self.breadth_window
                    and len(spy_hist) >= self.breadth_window):
                rsp_close = rsp_hist["close"].dropna()
                spy_close = spy_hist["close"].dropna()
                if (len(rsp_close) >= self.breadth_window
                        and len(spy_close) >= self.breadth_window):
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.breadth_window] - 1.0)
                    broad_participation = rsp_ret > spy_ret
        except Exception:
            pass

        target: dict[str, float] = {}

        if not broad_participation:
            # Narrow leadership / large-cap led — defensive
            for sym, w in [("SPY", 0.70), ("TLT", 0.27)]:
                if sym in closes_now.index:
                    target[sym] = w * self.exposure
        else:
            # Broad participation — rank sector ETFs by momentum
            need = self.sector_mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.sector_mom_window:
                # fallback
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure
            else:
                scores: dict[str, float] = {}
                for sym in SECTOR_ETFS:
                    if sym not in prices.columns:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.sector_mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.sector_mom_window] - 1.0)
                    if np.isfinite(ret):
                        scores[sym] = ret

                if not scores:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    longs = ranked[:min(self.top_k, len(ranked))]
                    per_w = self.exposure / len(longs)
                    for sym in longs:
                        target[sym] = per_w

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


NAME = "rsp_breadth_sector_tilt"
HYPOTHESIS = (
    "RSP/SPY breadth-based sector tilt: when equal-weight RSP outperforms cap-weight SPY on 30d "
    "return (broad market participation) hold top-2 sector ETFs by 42d momentum; "
    "else hold SPY+TLT 70/27; weekly rebalance; breadth drives sector choice not VIX or credit"
)

UNIVERSE = ["RSP", "SPY", "TLT"] + SECTOR_ETFS

STRATEGY = RspBreadthSectorTilt()
