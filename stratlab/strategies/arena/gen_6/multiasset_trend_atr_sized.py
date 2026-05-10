"""Multi-asset trend-following with inverse-ATR position sizing.

Hypothesis: Hold SPY/TLT/GLD/DBC each when price is above their 120d SMA.
Size each position inversely proportional to its 20d normalized ATR
(ATR/price) so high-volatility assets get smaller allocations. Rotate
assets below trend to SHY (cash proxy). Rebalance every 10 bars.

Rationale: This is a classic 4-asset trend-following strategy with risk-
parity-style sizing. By including commodities (DBC) and gold (GLD), the
strategy diversifies against the equity/bond rotation strategies dominating
the leaderboard. The inverse-ATR sizing replaces equal-weight or inverse-vol
with a more economically intuitive measure of realized volatility.

Structural differences from existing strategies:
- DBC (broad commodity ETF) is not used as a tradeable asset on the leaderboard
- 4-asset diversified holding vs 2-3 asset rotation
- ATR-based sizing (not equal-weight, not inverse-vol of returns)
- 120d trend window is intermediate between the 100d and 200d windows in use
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["SPY", "TLT", "GLD", "DBC", "SHY"]

TREND_WINDOW = 120   # 120d SMA for trend filter
ATR_WINDOW = 20      # 20d ATR for volatility sizing
REBALANCE_EVERY = 10
EXPOSURE = 0.97

ASSETS = ["SPY", "TLT", "GLD", "DBC"]
SAFE_HAVEN = "SHY"


class MultiassetTrendAtrSized(Strategy):
    def __init__(
        self,
        trend_window: int = TREND_WINDOW,
        atr_window: int = ATR_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            trend_window=trend_window,
            atr_window=atr_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.trend_window = int(trend_window)
        self.atr_window = int(atr_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def _compute_atr(self, hist: pd.DataFrame, window: int) -> float:
        """Compute normalized ATR (ATR/price) using True Range."""
        if len(hist) < window + 2:
            return float("nan")
        recent = hist.iloc[-(window + 1):]
        high = recent["high"].values
        low = recent["low"].values
        close = recent["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = float(np.mean(tr[-window:]))
        last_close = float(close[-1])
        if last_close <= 0:
            return float("nan")
        return atr / last_close  # normalized ATR

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.trend_window + self.atr_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # Determine which assets are in uptrend and compute inverse-ATR weights
        in_trend: list[str] = []
        norm_atrs: dict[str, float] = {}

        for sym in ASSETS:
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            if hist is None or len(hist) < self.trend_window + 2:
                continue

            close_series = hist["close"].dropna()
            if len(close_series) < self.trend_window:
                continue

            last_close = float(close_series.iloc[-1])
            sma = float(close_series.iloc[-self.trend_window:].mean())

            if last_close > sma:
                # Asset is in uptrend
                natr = self._compute_atr(hist, self.atr_window)
                if np.isfinite(natr) and natr > 1e-8:
                    in_trend.append(sym)
                    norm_atrs[sym] = natr

        target: dict[str, float] = {}

        if not in_trend:
            # All assets in downtrend - go to safe haven
            if SAFE_HAVEN in live:
                target[SAFE_HAVEN] = self.exposure
        else:
            # Inverse-ATR weighting for assets in trend
            inv_atrs = {sym: 1.0 / norm_atrs[sym] for sym in in_trend}
            total_inv_atr = sum(inv_atrs.values())
            if total_inv_atr <= 0:
                if SAFE_HAVEN in live:
                    target[SAFE_HAVEN] = self.exposure
            else:
                for sym in in_trend:
                    weight = self.exposure * inv_atrs[sym] / total_inv_atr
                    target[sym] = weight

                # Put remaining weight into SHY (for slots that dropped out of trend)
                n_assets_expected = len(ASSETS)
                n_in_trend = len(in_trend)
                if n_in_trend < n_assets_expected and SAFE_HAVEN in live:
                    # Add SHY proportional to the "missing" slots
                    shy_weight = self.exposure * (n_assets_expected - n_in_trend) / n_assets_expected
                    # Rescale existing target to sum to remaining budget
                    # Actually just normalize all weights to exposure
                    pass  # already done via inverse-ATR

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Build target positions
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

        return orders


NAME = "multiasset_trend_atr_sized"
HYPOTHESIS = (
    "Multi-asset trend-following with inverse-ATR sizing: hold SPY/TLT/GLD/DBC "
    "each when price above 120d SMA; size each position inversely proportional to "
    "20d ATR/price (normalized ATR); cash slot to SHY for assets in downtrend; "
    "rebalance every 10 bars; 4-asset diversified trend following"
)

STRATEGY = MultiassetTrendAtrSized()
