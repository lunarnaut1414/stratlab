"""Gold-to-bond ratio macro regime strategy — gen_6 sonnet-7

Hypothesis: Use the GLD/TLT price ratio to identify macro regime:
  - When GLD/TLT 20d MA > 60d MA (gold outperforming bonds = inflation/
    risk-on): hold SPY 60% + XLE 37% (inflation beneficiary tilt)
  - When GLD/TLT 20d MA < 60d MA AND GLD/TLT 63d return < 0 (gold
    falling vs bonds = deflationary/risk-off): hold TLT 97%
  - Neutral (GLD/TLT cross bearish but not confirmed): hold SPY 97%
  Rebalance weekly.

Rationale:
  The GLD/TLT ratio captures real rate expectations — when gold outperforms
  bonds, real rates are falling (inflation rising faster than nominal), which
  is risk-on for equities and especially cyclicals/energy. When bonds
  outperform gold, real rates are rising or deflationary pressure builds,
  which is risk-off. This signal is economically distinct from VIX (equity
  fear), credit spreads (credit risk), or SPY trend (price momentum).

  Distinct from existing leaderboard:
  - GLD/TLT ratio as primary regime signal (not used in any existing strategy)
  - XLE (energy) as risk-on allocation alongside SPY (not standard)
  - Inflation-aware cross-asset positioning
  - Weekly rebalance with 20d/60d MA crossover
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

FAST_MA = 20
SLOW_MA = 60
CONFIRM_WINDOW = 63    # additional 63d return confirmation for risk-off
REBALANCE_EVERY = 5    # weekly
EXPOSURE = 0.97


class GoldBondRegime(Strategy):
    """GLD/TLT ratio regime: SPY+XLE in inflation regimes, TLT in deflation."""

    def __init__(
        self,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        confirm_window: int = CONFIRM_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            confirm_window=confirm_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.confirm_window = int(confirm_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.slow_ma + self.confirm_window + 5
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

        # Compute GLD/TLT ratio signal
        gld_tlt_ratio_series = None
        try:
            gld_hist = ctx.history("GLD")
            tlt_hist = ctx.history("TLT")
            need = self.slow_ma + self.confirm_window + 5
            if len(gld_hist) >= need and len(tlt_hist) >= need:
                gld_close = gld_hist["close"].dropna().values
                tlt_close = tlt_hist["close"].dropna().values
                min_len = min(len(gld_close), len(tlt_close))
                gld_close = gld_close[-min_len:]
                tlt_close = tlt_close[-min_len:]
                # Compute ratio: aligned by same index
                ratio = gld_close / tlt_close
                gld_tlt_ratio_series = ratio
        except Exception:
            pass

        target: dict[str, float] = {}

        if gld_tlt_ratio_series is None or len(gld_tlt_ratio_series) < self.slow_ma + 2:
            # Fallback: hold SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        else:
            ratio = gld_tlt_ratio_series
            fast_ma = float(np.mean(ratio[-self.fast_ma:]))
            slow_ma = float(np.mean(ratio[-self.slow_ma:]))

            gld_tlt_bullish = fast_ma > slow_ma

            # Additional confirmation: 63d return of ratio
            if len(ratio) >= self.confirm_window + 2:
                ratio_63d_ret = float(ratio[-1] / ratio[-self.confirm_window] - 1.0)
            else:
                ratio_63d_ret = 0.0

            if gld_tlt_bullish:
                # Risk-on / inflation regime: SPY 60% + XLE 37%
                if "SPY" in closes_now.index:
                    target["SPY"] = 0.60 * self.exposure
                if "XLE" in closes_now.index:
                    target["XLE"] = 0.37 * self.exposure
            elif not gld_tlt_bullish and ratio_63d_ret < -0.03:
                # Deflationary / risk-off: TLT 97%
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            else:
                # Neutral: SPY 97%
                if "SPY" in closes_now.index:
                    target["SPY"] = self.exposure

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


NAME = "gold_bond_regime"
HYPOTHESIS = (
    "Gold-to-bond (GLD/TLT) ratio regime: GLD/TLT 20d MA > 60d MA → SPY 60%+XLE 37% "
    "(inflation/risk-on); GLD/TLT bearish AND 63d ratio return < -3% → TLT 97% (deflation); "
    "neutral → SPY 97%; weekly rebalance; cross-asset real-rate regime signal"
)
UNIVERSE = ["SPY", "TLT", "GLD", "XLE"]
STRATEGY = GoldBondRegime()
