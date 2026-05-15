"""Realized Volatility Compression Regime Signal — gen_9 sonnet-8

Hypothesis:
When SPY's 10-day realized vol is compressed vs its 63-day realized vol
(vol compression = calm trending regime), hold QQQ 97%.
When vol expands (10d >> 63d vol), rotate defensive.
SPY 200d SMA outer bear gate.

Regime classification:
  - vol_ratio = SPY 10d realized vol / SPY 63d realized vol
  - ratio < 0.70 (vol compression, calm/trending): QQQ 97%
  - ratio > 1.30 (vol expansion, turbulent): TLT 97%
  - neutral (0.70 - 1.30) AND SPY above 200d SMA: SPY 97%
  - SPY below 200d SMA: TLT 97% (outer bear gate, overrides all)
Rebalance every 5 bars (weekly).

Rationale:
Realized vol compression (short-window vol below long-window vol) is a classic
regime signal — it identifies periods of sustained low volatility that tend to
precede and accompany trending bull markets. In these windows, QQQ's beta
amplification is a feature, not a bug. Vol expansion flips the logic: high
short-term vol relative to long-term baseline signals regime stress, where
bonds outperform. This signal is distinct from VIX level (which measures
implied vol), credit spreads, yield curves, and all other macro signals.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
SHORT_VOL_WINDOW = 10     # short realized vol lookback
LONG_VOL_WINDOW = 63      # long realized vol baseline
TREND_WINDOW = 200        # SPY 200d SMA bear gate
COMPRESS_THRESHOLD = 0.70  # vol ratio below this -> QQQ (compression)
EXPAND_THRESHOLD = 1.30    # vol ratio above this -> TLT (expansion)
EXPOSURE = 0.97


class VolCompressionQqQSpy(Strategy):
    """Realized vol compression routing QQQ/SPY/TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_vol_window: int = SHORT_VOL_WINDOW,
        long_vol_window: int = LONG_VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        compress_threshold: float = COMPRESS_THRESHOLD,
        expand_threshold: float = EXPAND_THRESHOLD,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_vol_window=short_vol_window,
            long_vol_window=long_vol_window,
            trend_window=trend_window,
            compress_threshold=compress_threshold,
            expand_threshold=expand_threshold,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_vol_window = int(short_vol_window)
        self.long_vol_window = int(long_vol_window)
        self.trend_window = int(trend_window)
        self.compress_threshold = float(compress_threshold)
        self.expand_threshold = float(expand_threshold)
        self.exposure = float(exposure)

    def _realized_vol(self, close_series: "pd.Series", window: int) -> float:  # type: ignore[name-defined]
        """Compute annualized realized vol over last `window` bars."""
        if len(close_series) < window + 1:
            return float("nan")
        tail = close_series.iloc[-(window + 1):]
        log_rets = np.log(tail.values[1:] / tail.values[:-1])
        if len(log_rets) == 0:
            return float("nan")
        return float(np.std(log_rets, ddof=1)) * (252 ** 0.5)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.long_vol_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []

        live = {s: float(closes_now[s]) for s in closes_now.index
                if float(closes_now[s]) > 0}

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY bear gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 2:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        if not spy_bull:
            target = {"TLT": self.exposure}
        else:
            # --- Vol compression signal ---
            short_vol = self._realized_vol(spy_close, self.short_vol_window)
            long_vol = self._realized_vol(spy_close, self.long_vol_window)

            if not np.isfinite(short_vol) or not np.isfinite(long_vol) or long_vol <= 1e-10:
                target = {"SPY": self.exposure}
            else:
                vol_ratio = short_vol / long_vol
                if vol_ratio < self.compress_threshold:
                    # Vol compression: QQQ
                    target = {"QQQ": self.exposure}
                elif vol_ratio > self.expand_threshold:
                    # Vol expansion: TLT
                    target = {"TLT": self.exposure}
                else:
                    # Neutral: SPY
                    target = {"SPY": self.exposure}

        # --- Build orders ---
        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
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


NAME = "gen9_vol_compression_qqq_spy"
HYPOTHESIS = (
    "Realized vol compression: SPY 10d/63d vol ratio < 0.70 -> QQQ 97%; "
    "ratio > 1.30 -> TLT 97%; neutral AND SPY bull -> SPY 97%; "
    "SPY bear -> TLT 97%; weekly rebalance."
)

UNIVERSE = ["QQQ", "SPY", "TLT"]

STRATEGY = VolCompressionQqQSpy()
