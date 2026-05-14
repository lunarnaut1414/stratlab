"""SP500 composite momentum with credit-tilted factor selection.

Hypothesis: Always-invested SP500 strategy (bear-market TLT gate only) that
uses JNK credit signal to select WHICH TYPE of SP500 stocks to hold:

When JNK is above its 30d SMA by >0.5% (credit risk-on):
  Hold top-10 SP500 stocks by composite momentum (21d + 63d + 126d average return),
  equal-weight — concentrated bet on strongest multi-horizon momentum names.

When JNK is within ±0.5% of its 30d SMA (credit neutral):
  Hold top-20 SP500 stocks by composite momentum, inverse-vol weighted —
  broader diversification with risk adjustment.

When JNK is below 30d SMA by >0.5% (credit risk-off, but equity bull):
  Hold top-20 SP500 stocks by LOWEST 21d realized vol, equal-weight —
  stays in equities but defensive within the asset class.

Bear market gate: When SPY below 200d SMA, rotate to TLT regardless.
Rebalance every 10 bars.

Rationale: The key innovation is the credit signal tilts the FACTOR EXPOSURE
within SP500 — not between SP500 and bonds (as most strategies do). In
credit-strong regimes, momentum is reliable. In credit-neutral regimes,
inverse-vol sizing adds safety without abandoning equity. In credit-weak
regimes (but still in bull market), low-vol factor provides defensive positioning.

Distinction from existing:
  - All credit-gated SP500 strategies on leaderboard rotate to TLT/GLD/IEF on credit-off
  - This stays in SP500 stocks in all three regimes (only TLT in true bear market)
  - Composite 3-horizon momentum score not used by any existing strategy
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10     # bars (~2 weeks)
TREND_WINDOW = 200       # SPY bear market gate
JNK_MA = 30              # JNK short MA for credit regime
JNK_BAND = 0.005         # ±0.5% band around MA for neutral zone
MOM_WINDOWS = (21, 63, 126)  # composite momentum lookbacks
VOL_WINDOW = 21          # realized vol window
TOP_K_AGGRESSIVE = 10    # holdings in risk-on
TOP_K_NEUTRAL = 20       # holdings in neutral
TOP_K_DEFENSIVE = 20     # holdings in defensive (low-vol within equity)
STOCK_TREND_WINDOW = 100 # stock-level trend filter for defensive mode
EXPOSURE = 0.97


class SP500CompositeMomCreditTilt(Strategy):
    """Credit-signal-tilted SP500 factor rotation: momentum vs low-vol within equities."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        jnk_ma: int = JNK_MA,
        jnk_band: float = JNK_BAND,
        vol_window: int = VOL_WINDOW,
        top_k_aggressive: int = TOP_K_AGGRESSIVE,
        top_k_neutral: int = TOP_K_NEUTRAL,
        top_k_defensive: int = TOP_K_DEFENSIVE,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            jnk_ma=jnk_ma,
            jnk_band=jnk_band,
            vol_window=vol_window,
            top_k_aggressive=top_k_aggressive,
            top_k_neutral=top_k_neutral,
            top_k_defensive=top_k_defensive,
            stock_trend_window=stock_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.jnk_ma = int(jnk_ma)
        self.jnk_band = float(jnk_band)
        self.vol_window = int(vol_window)
        self.top_k_aggressive = int(top_k_aggressive)
        self.top_k_neutral = int(top_k_neutral)
        self.top_k_defensive = int(top_k_defensive)
        self.stock_trend_window = int(stock_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, max(MOM_WINDOWS)) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY bear market gate
        try:
            spy_hist = ctx.history("SPY")
        except Exception:
            return []
        if spy_hist is None or len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear market: TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Determine credit regime
            credit_regime = "neutral"
            try:
                jnk_hist = ctx.history("JNK")
                if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 5:
                    jnk_close = jnk_hist["close"].dropna()
                    if len(jnk_close) >= self.jnk_ma:
                        jnk_current = float(jnk_close.iloc[-1])
                        jnk_ma_val = float(jnk_close.iloc[-self.jnk_ma:].mean())
                        ratio = (jnk_current - jnk_ma_val) / jnk_ma_val
                        if ratio > self.jnk_band:
                            credit_regime = "risk_on"
                        elif ratio < -self.jnk_band:
                            credit_regime = "risk_off"
            except Exception:
                pass

            need = max(MOM_WINDOWS) + 5
            prices = ctx.closes_window(need)
            if len(prices) < max(MOM_WINDOWS):
                return []

            if credit_regime == "risk_off":
                # Defensive within equity: lowest realized vol above 100d SMA
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < max(self.vol_window, self.stock_trend_window) + 2:
                        continue
                    # Stock trend filter
                    if len(col) >= self.stock_trend_window:
                        sma = float(col.iloc[-self.stock_trend_window:].mean())
                        if float(col.iloc[-1]) < sma:
                            continue
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue
                    scores[sym] = -rv  # lower vol = higher score

                k = self.top_k_defensive
                if len(scores) < 5:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    k = min(k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    per_weight = self.exposure / len(ranked)
                    for sym in ranked:
                        target[sym] = per_weight
            else:
                # Risk-on or neutral: composite momentum ranking
                # Composite score = average return across 21d, 63d, 126d windows
                scores = {}
                inv_vols: dict[str, float] = {}
                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < max(MOM_WINDOWS) + 2:
                        continue
                    rets = []
                    for w in MOM_WINDOWS:
                        if len(col) >= w + 1:
                            p_end = float(col.iloc[-1])
                            p_start = float(col.iloc[-w])
                            if p_start > 0 and np.isfinite(p_start) and np.isfinite(p_end):
                                rets.append(p_end / p_start - 1.0)
                    if len(rets) < len(MOM_WINDOWS):
                        continue
                    composite = float(np.mean(rets))
                    if not np.isfinite(composite):
                        continue
                    scores[sym] = composite

                    # Inverse-vol for neutral regime weighting
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) >= self.vol_window + 1:
                        logr = np.log(tail.values[1:] / tail.values[:-1])
                        rv = float(np.std(logr))
                        if rv > 1e-6 and np.isfinite(rv):
                            inv_vols[sym] = 1.0 / rv

                if credit_regime == "risk_on":
                    k = self.top_k_aggressive
                    if len(scores) < 5:
                        if "SPY" in closes_now.index:
                            target["SPY"] = self.exposure
                    else:
                        k = min(k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                        per_weight = self.exposure / len(ranked)
                        for sym in ranked:
                            target[sym] = per_weight
                else:
                    # Neutral: inverse-vol weighted top-20
                    k = self.top_k_neutral
                    if len(scores) < 5:
                        if "SPY" in closes_now.index:
                            target["SPY"] = self.exposure
                    else:
                        k = min(k, len(scores))
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                        iv_sum = sum(inv_vols.get(s, 1.0) for s in ranked)
                        if iv_sum <= 0:
                            iv_sum = len(ranked)
                        for sym in ranked:
                            w = inv_vols.get(sym, 1.0) / iv_sum
                            target[sym] = self.exposure * w

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "JNK", "SPY"]


NAME = "sp500_composite_mom_credit_tilt"
HYPOTHESIS = (
    "SP500 tri-state factor rotation via JNK credit signal: credit strong -> top-10 by composite "
    "21d+63d+126d momentum equal-weight; credit neutral -> top-20 inverse-vol weighted; "
    "credit weak (but equity bull) -> top-20 lowest-21d-vol stocks above 100d SMA; "
    "SPY 200d gate to TLT; bi-weekly rebalance"
)

UNIVERSE = _universe

STRATEGY = SP500CompositeMomCreditTilt()
