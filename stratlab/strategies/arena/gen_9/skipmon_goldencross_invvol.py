"""gen_9 sonnet-3 — Skip-month Momentum with Golden Cross Gate, Per-Stock 50d SMA, Inverse-Vol

Hypothesis: Combine the best OOS-validated features from gen7 and gen8:
  - Skip-month momentum (126d skip 21d, Jegadeesh-Titman) — best OOS gen8 (0.63)
  - SPY 50d vs 150d golden cross gate — best OOS gen7 (0.72)
  - Per-stock 50d SMA trend filter — from best OOS gen7
  - Inverse-vol weighting — from best OOS gen8
  - Hold top-20 SP500 stocks; IEF defensive; biweekly rebalance

Rationale: gen7_sp500_126d_stock_50sma_goldencross (0.72 OOS) used golden cross +
per-stock 50d SMA + equal weight + 126d straight momentum.
gen8_sp500_skipmon_63sma_momentum (0.63 OOS) used skip-month + per-stock 63d SMA + inv-vol.
No strategy has combined ALL FOUR features. The combination avoids:
- Short-term reversal (skip-month)
- False trend signals (per-stock 50d SMA)
- SPY bear entry (golden cross gate)
- Concentration in high-vol names (inverse-vol weighting)

Differences from existing: gen7_goldencross uses 126d straight (not skip), equal weight.
gen8_skipmon uses 200d SPY SMA (not golden cross), 63d stock SMA.
This strategy uses 50d/150d golden cross (faster than 200d) + skip-month + per-stock 50d + inv-vol.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10    # biweekly
MOM_LONG = 126          # total lookback for skip-month
MOM_SKIP = 21           # skip most recent 21 days
FAST_MA = 50            # SPY golden cross fast MA
SLOW_MA = 150           # SPY golden cross slow MA
STOCK_SMA = 50          # per-stock trend filter
VOL_WINDOW = 21         # realized vol for inverse-vol weighting
TOP_K = 20
EXPOSURE = 0.97
_SPY = "SPY"
_IEF = "IEF"


class SkipmonGoldencrossInvvol(Strategy):
    """Skip-month momentum with SPY golden cross gate, per-stock 50d SMA, inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        stock_sma: int = STOCK_SMA,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            stock_sma=stock_sma,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.stock_sma = int(stock_sma)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.mom_long, self.stock_sma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY golden cross gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.slow_ma + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.slow_ma:
            return []
        spy_fast_sma = float(spy_close.iloc[-self.fast_ma:].mean())
        spy_slow_sma = float(spy_close.iloc[-self.slow_ma:].mean())
        golden_cross = spy_fast_sma > spy_slow_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not golden_cross:
            # SPY death cross — defensive IEF
            if _IEF in live:
                target[_IEF] = self.exposure
        else:
            # Golden cross — select top-K SP500 stocks via skip-month momentum
            need = self.mom_long + 10
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Compute skip-month momentum: 126d return skipping most recent 21d
                # price at bar [-mom_skip - 1] vs price at bar [-mom_long - 1]
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _IEF):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue
                    # Skip-month: return from mom_long bars ago to mom_skip bars ago
                    p_end = float(col.iloc[-(self.mom_skip + 1)])
                    p_start = float(col.iloc[-(self.mom_long + 1)] if len(col) > self.mom_long else col.iloc[0])
                    if p_start <= 0:
                        continue
                    ret = p_end / p_start - 1.0
                    if np.isfinite(ret):
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                    # Apply per-stock 50d SMA filter AND compute inverse-vol weights
                    inv_vols: dict[str, float] = {}
                    count = 0
                    for sym in ranked:
                        if count >= self.top_k:
                            break
                        # Per-stock 50d SMA filter
                        try:
                            s_hist = ctx.history(sym)
                        except KeyError:
                            continue
                        if len(s_hist) < self.stock_sma + 2:
                            continue
                        s_close = s_hist["close"].dropna()
                        if len(s_close) < self.stock_sma:
                            continue
                        stock_sma_val = float(s_close.iloc[-self.stock_sma:].mean())
                        stock_price = float(s_close.iloc[-1])
                        if stock_price <= stock_sma_val:
                            continue  # below own 50d SMA — skip

                        # Compute 21d realized vol for inverse-vol weighting
                        if len(s_close) >= self.vol_window + 1:
                            rets = s_close.iloc[-(self.vol_window + 1):].pct_change().dropna()
                            rv = float(rets.std()) * np.sqrt(252)
                        else:
                            rv = 0.20  # fallback vol
                        if rv <= 0:
                            rv = 0.20
                        if sym in live:
                            inv_vols[sym] = 1.0 / rv
                            count += 1

                    if not inv_vols:
                        # No qualifying stocks — fall back to SPY
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        total_invvol = sum(inv_vols.values())
                        for sym, iv in inv_vols.items():
                            target[sym] = self.exposure * iv / total_invvol

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_IEF, _SPY]


UNIVERSE = _universe

NAME = "skipmon_goldencross_invvol"
HYPOTHESIS = (
    "Skip-month momentum (126d-skip-21d) with SPY golden-cross gate (50d vs 150d) AND "
    "per-stock 50d SMA filter AND inverse-vol weighting: combines best OOS features from "
    "gen7 (golden cross gate + per-stock 50d) and gen8 (skip-month + stock SMA filter + inv-vol); "
    "hold top-20 SP500 stocks; IEF defensive; biweekly rebalance"
)

STRATEGY = SkipmonGoldencrossInvvol()
