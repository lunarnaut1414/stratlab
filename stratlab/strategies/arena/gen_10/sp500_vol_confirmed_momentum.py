"""SP500 momentum with volume-expansion quality filter.

Hypothesis:
    Price momentum backed by rising volume is more sustainable than momentum on
    declining volume. Declining volume during a price run is a classic technical
    warning sign (distribution) — institutions selling into retail buying.

    This strategy adds a VOLUME CONFIRMATION quality screen to standard SP500
    momentum: only hold stocks where recent trading volume is expanding (20d
    average volume > 63d average volume), confirming that the momentum is
    supported by participation, not thinning out.

    Mechanism:
    - 20d avg volume / 63d avg volume > 1.0 means recent volume is above
      the medium-term average → momentum backed by rising participation
    - This is analogous to the RSI > 35 filter (gen9_sp500_rsi_quality_momentum)
      but uses VOLUME rather than price oscillator as the quality signal
    - Volume confirmation is structurally different from both RSI and SMA-distance
      filters used in other strategies

Design:
    - Compute 20d and 63d average daily volume for each SP500 stock
    - Only include stocks with vol_ratio (20d/63d) > 1.0 (volume expanding)
    - Rank qualifying stocks by 126d total return (momentum)
    - Hold top-15 inverse-vol weighted
    - SPY 200d SMA outer gate: defensive IEF in bear markets
    - Biweekly rebalance (10 bars)

Differentiation from leaderboard:
    - gen9_sp500_rsi_quality_momentum: RSI > 35 quality filter (price oscillator)
    - gen10_sp500_sma_distance_momentum: SMA-distance band filter (price level)
    - This strategy uses VOLUME EXPANSION as the quality signal — entirely different
      data dimension (trading activity vs price structure)
    - Volume data is in the history() context for all stocks; no special access needed
    - Expected lower corr to existing strategies because the stock universe selected
      by volume expansion differs meaningfully from price-based filters
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOMENTUM_WINDOW = 126       # 6-month momentum
VOL_SHORT = 20              # short-term avg volume lookback
VOL_LONG = 63               # longer-term avg volume for comparison
INV_VOL_WINDOW = 21         # realized vol for position sizing
SPY_TREND_WINDOW = 200      # outer bear gate
TOP_K = 15                  # max positions
EXPOSURE = 0.97
VOL_RATIO_THRESHOLD = 1.0   # 20d avg vol must exceed 63d avg vol


class SP500VolConfirmedMomentum(Strategy):
    """SP500 126d momentum with volume-expansion quality filter (20d > 63d avg vol).

    Volume confirmation selects stocks with expanding participation; inverse-vol weighted;
    SPY 200d gate to IEF defensive; biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        vol_short: int = VOL_SHORT,
        vol_long: int = VOL_LONG,
        inv_vol_window: int = INV_VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        vol_ratio_threshold: float = VOL_RATIO_THRESHOLD,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            vol_short=vol_short,
            vol_long=vol_long,
            inv_vol_window=inv_vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
            vol_ratio_threshold=vol_ratio_threshold,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.vol_short = int(vol_short)
        self.vol_long = int(vol_long)
        self.inv_vol_window = int(inv_vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.vol_ratio_threshold = float(vol_ratio_threshold)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.momentum_window, self.vol_long) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer gate
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
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = self.exposure
        else:
            # Need enough data for momentum + volume
            need = max(self.momentum_window, self.vol_long) + self.inv_vol_window + 5

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            # Iterate over tradeable symbols, loading per-stock history
            for sym in ctx.symbols:
                try:
                    sym_hist = ctx.history(sym)
                except KeyError:
                    continue

                if len(sym_hist) < need:
                    continue

                close_col = sym_hist["close"].dropna()
                n = len(close_col)
                if n < need:
                    continue

                # Volume confirmation filter
                if "volume" in sym_hist.columns:
                    vol_col = sym_hist["volume"].dropna()
                    if len(vol_col) >= self.vol_long + 2:
                        vol_short_avg = float(vol_col.iloc[-self.vol_short:].mean())
                        vol_long_avg = float(vol_col.iloc[-self.vol_long:].mean())
                        if vol_long_avg <= 0 or not np.isfinite(vol_long_avg):
                            continue
                        vol_ratio = vol_short_avg / vol_long_avg
                        if vol_ratio < self.vol_ratio_threshold:
                            continue  # Volume declining — skip
                    else:
                        continue  # Not enough volume data
                else:
                    continue  # No volume data — skip

                # 126d momentum
                if n < self.momentum_window + 2:
                    continue
                p_end = float(close_col.iloc[-1])
                p_start = float(close_col.iloc[-self.momentum_window])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight (realized vol)
                tail = close_col.values[-(self.inv_vol_window + 1):]
                if len(tail) < self.inv_vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                if "IEF" in closes_now.index:
                    target["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target weights
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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


NAME = "sp500_vol_confirmed_momentum"
HYPOTHESIS = (
    "SP500 126d momentum with volume-expansion quality filter: only rank stocks where "
    "20d average volume > 63d average volume (volume expanding, momentum confirmed by "
    "participation); hold top-15 inverse-vol weighted; SPY 200d outer gate to IEF "
    "defensive; biweekly rebalance — volume confirmation is a distinct quality dimension "
    "from RSI, SMA-distance, or near-52wk-high filters used in existing strategies"
)

UNIVERSE = _universe

STRATEGY = SP500VolConfirmedMomentum()
