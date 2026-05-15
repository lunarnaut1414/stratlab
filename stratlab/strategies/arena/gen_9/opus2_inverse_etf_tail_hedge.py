"""opus-2 gap_finder: Inverse-ETF tail-hedge sleeve.

Gap identified: every defensive routing in gen_7-9 uses TLT/SHY/IEF as the
non-equity sleeve. NO strategy uses inverse ETFs (SH=inverse SPY, PSQ=inverse
QQQ) as an *active hedge* while keeping core equity allocation. This is
structurally different from rotating into bonds — it's a beta-neutralizing
overlay on top of always-on equity. SH covers IS (since 2006); PSQ also covers.

Hypothesis: When VIX>22 or QQQ below 200d SMA, keep 80% QQQ but add 15% SH
(short-SPY exposure) to neutralize ~half of beta during stress, while
preserving equity exposure for the rebound. In calm regimes (VIX<=22 and QQQ
above 200d SMA), drop SH entirely and run 95% QQQ. This tests whether
*explicit hedging* beats *defensive routing* — the existing leaderboard has
0 strategies using this construction.

Mechanics:
  - Universe: QQQ + SPY + SH + ^VIX (signal-only)
  - Hedge trigger: VIX>22 OR QQQ_close < QQQ_200d_SMA
  - Hedged state:  80% QQQ + 15% SH (net ~65% long equity)
  - Calm state:    95% QQQ
  - Rebalance every 5 bars

Note: SH carries borrow drag (~1% annually) and tracking error, so this is a
genuine performance test — not a free hedge. The strategy must earn enough
calm-state upside to offset the hedge cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["QQQ", "SPY", "SH", "^VIX"]

VIX_THRESHOLD = 22.0
SMA_PERIOD = 200
REBALANCE_EVERY = 5


class InverseEtfTailHedge(Strategy):
    def __init__(
        self,
        vix_threshold: float = VIX_THRESHOLD,
        sma_period: int = SMA_PERIOD,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            vix_threshold=vix_threshold,
            sma_period=sma_period,
            rebalance_every=rebalance_every,
        )
        self.vix_threshold = float(vix_threshold)
        self.sma_period = int(sma_period)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.sma_period + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # VIX level
        vix_level = float("nan")
        try:
            vix_hist = ctx.history("^VIX")
            if vix_hist is not None and len(vix_hist) >= 1:
                vix_level = float(vix_hist["close"].iloc[-1])
        except Exception:
            pass

        # QQQ trend
        try:
            qqq_hist = ctx.history("QQQ")
        except KeyError:
            return []
        if qqq_hist is None or len(qqq_hist) < self.sma_period + 1:
            return []
        qc = qqq_hist["close"].dropna()
        qqq_now = float(qc.iloc[-1])
        qqq_sma = float(qc.iloc[-self.sma_period:].mean())
        qqq_below = np.isfinite(qqq_now) and np.isfinite(qqq_sma) and qqq_now < qqq_sma
        vix_high = np.isfinite(vix_level) and vix_level > self.vix_threshold

        if vix_high or qqq_below:
            # Hedge sleeve active
            target = {"QQQ": 0.80, "SH": 0.15}
        else:
            target = {"QQQ": 0.95}

        live = ctx.closes()
        if live.empty:
            return []
        live_dict = {s: float(p) for s, p in live.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live_dict)
        if equity <= 0:
            return []

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))

        for sym, weight in target.items():
            price = live_dict.get(sym)
            if price is None or price <= 0:
                continue
            target_shares = int(equity * weight / price)
            cur_shares = int(ctx.position(sym).size)
            delta = target_shares - cur_shares
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))
        return orders


NAME = "opus2_inverse_etf_tail_hedge"
HYPOTHESIS = (
    "Inverse-ETF tail-hedge sleeve: in calm regime (VIX<=22 AND QQQ>200d SMA) hold 95% QQQ; "
    "in stress (VIX>22 OR QQQ<200d SMA) hold 80% QQQ + 15% SH (inverse SPY) as an active hedge "
    "rather than rotating into bonds. Tests explicit hedging vs defensive routing — untouched after 4 rounds."
)

STRATEGY = InverseEtfTailHedge()
