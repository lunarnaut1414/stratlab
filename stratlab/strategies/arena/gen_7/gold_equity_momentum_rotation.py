"""Gold-equity momentum rotation: hold GLD vs QQQ based on 63d relative momentum.

Hypothesis: Compare GLD and SPY on 63-day total return:
  - When GLD outperforms SPY (GLD 63d > SPY 63d) AND GLD is above its 200d SMA:
    hold GLD 97% — gold leading signals risk-off / inflation regime
  - When SPY outperforms GLD AND QQQ is above its 200d SMA:
    hold QQQ 97% — equity momentum with tech concentration in bull markets
  - When both are below their respective 200d SMAs:
    hold IEF 97% — dual bear market in equities AND gold is rare but severe

Rebalance every 5 bars (weekly) with a 3-bar minimum hold.

Rationale: Gold and equities tend to alternate leadership based on macro regime.
In reflation / uncertainty periods (2010-2012, crisis), gold leads. In pure risk-on
bull markets (2013-2018), equities lead strongly. The 63-day relative momentum
cleanly captures these regime shifts. Using QQQ (not SPY) in equity-leading
regimes concentrates in the highest-momentum sector. The 200d SMA gate on both
instruments prevents holding a collapsing asset.

Distinction from existing strategies:
  - No existing strategy uses GLD vs equity relative momentum as the primary signal
  - gen5_risk_parity_spy_tlt_gld always holds GLD; this rotates away from it
  - gen5_tech_vs_defensive_rotation uses XLK vs XLU, not gold vs equities
  - The gold/equity rotation signal is uncorrelated to all credit/VIX signals
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5      # bars (~1 week)
MIN_HOLD_BARS = 3        # minimum hold before switching
MOMENTUM_WINDOW = 63     # ~3 months for relative momentum comparison
TREND_WINDOW = 200       # 200d SMA for trend gate
EXPOSURE = 0.97


class GoldEquityMomentumRotation(Strategy):
    """Hold GLD when gold leads equities by 63d momentum; QQQ when equities lead; IEF in dual bear."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        min_hold_bars: int = MIN_HOLD_BARS,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            min_hold_bars=min_hold_bars,
            momentum_window=momentum_window,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.min_hold_bars = int(min_hold_bars)
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)
        self._last_rebal = -999

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []

        bars_since_rebal = ctx.idx - self._last_rebal
        if bars_since_rebal < self.min_hold_bars:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        need = self.trend_window + 5
        prices = ctx.closes_window(need)
        if len(prices) < self.momentum_window:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Compute 63d momentum and 200d SMA for GLD, SPY, QQQ
        def get_mom_and_trend(sym: str):
            """Return (63d_return, is_above_200d_sma) or (nan, False) if insufficient data."""
            if sym not in prices.columns:
                return float("nan"), False
            col = prices[sym].dropna()
            if len(col) < self.trend_window:
                return float("nan"), False
            sma200 = float(col.iloc[-self.trend_window:].mean())
            current = float(col.iloc[-1])
            above_trend = current > sma200
            if len(col) < self.momentum_window + 2:
                return float("nan"), above_trend
            p_start = float(col.iloc[-self.momentum_window])
            if p_start <= 0:
                return float("nan"), above_trend
            ret = current / p_start - 1.0
            if not np.isfinite(ret):
                return float("nan"), above_trend
            return ret, above_trend

        gld_ret, gld_above_trend = get_mom_and_trend("GLD")
        spy_ret, spy_above_trend = get_mom_and_trend("SPY")
        qqq_above_trend = get_mom_and_trend("QQQ")[1]

        # Decision logic
        target: dict[str, float] = {}

        gld_valid = np.isfinite(gld_ret)
        spy_valid = np.isfinite(spy_ret)

        if not gld_above_trend and not spy_above_trend:
            # Both gold and equity in downtrend — hold IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        elif gld_valid and spy_valid and gld_ret > spy_ret and gld_above_trend:
            # Gold outperforming and above trend — hold GLD
            if "GLD" in closes_now.index:
                target["GLD"] = self.exposure
        elif qqq_above_trend:
            # Equity regime leading — hold QQQ if it's above trend
            if "QQQ" in closes_now.index:
                target["QQQ"] = self.exposure
        elif spy_above_trend:
            # SPY above trend but QQQ below — hold SPY
            if "SPY" in closes_now.index:
                target["SPY"] = self.exposure
        elif gld_above_trend:
            # Only gold above trend
            if "GLD" in closes_now.index:
                target["GLD"] = self.exposure
        else:
            # Fallback: IEF
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure

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


NAME = "gold_equity_momentum_rotation"
HYPOTHESIS = (
    "Gold-equity momentum rotation: hold GLD when GLD 63d return > SPY 63d return AND GLD above 200d SMA "
    "(gold leadership = risk-off/inflation); hold QQQ when equities lead AND QQQ above 200d SMA; "
    "hold IEF when both below 200d SMA; weekly rebalance with min-3-bar hold"
)

UNIVERSE = ["GLD", "QQQ", "SPY", "IEF", "SHY"]

STRATEGY = GoldEquityMomentumRotation()
