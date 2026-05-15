"""SP500 skip-month momentum with per-stock trailing stop exits.

Hypothesis (sonnet-1, gen_10):
    Select top-15 SP500 stocks by 126d-skip-21d momentum (Jegadeesh-Titman).
    Inverse-vol weighted. SPY 200d SMA bear gate to IEF.
    Each stock has an independent 10% trailing stop from its peak since entry
    — exits a name early when it reverses, freeing capital for the next
    rebalance. Entry/refresh every 21 bars (monthly); exits can fire any bar.

Rationale:
  - gen9 best performers used vol-targeting (portfolio-level) or quality filters
    (per-stock). Trailing stops are a DIFFERENT risk-control mechanism: adaptive
    name-level exit that responds to realized drawdown, not vol forecast.
  - gen7_opus1_sp500_idio_trailstop combined trailing stops with idiosyncratic
    (beta-adjusted) momentum. This strategy uses RAW skip-month momentum for
    selection (different signal) — uncorrelated angle.
  - gen9_gen9_sp500_voltarget_skipmon: same skip-month selection but portfolio
    vol-targeting. This strategy: per-stock trailing stops instead. Different
    risk mechanism, different loss-mode profile.
  - Trailing stops are regime-invariant: they fire on realized drawdown,
    not on VIX level. OOS retention expected: HIGH.

Distinct:
  - Different from all vol-targeting strategies (portfolio-level vol target).
  - Different from gen7 idio+trailstop (different selection: raw skip-month
    vs beta-adjusted idiosyncratic momentum).
  - Per-stock trailing stop not present in any gen_10 strategy so far.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ENTRY_REFRESH = 21       # monthly: refresh positions / full rank
MOM_LOOKBACK = 126       # 6-month momentum
MOM_SKIP = 21            # skip last 1 month (Jegadeesh-Titman)
TREND_WINDOW = 200       # SPY 200d SMA gate
TOP_K = 15               # number of stocks to hold
VOL_WINDOW = 21          # for inverse-vol weights
TRAIL_STOP = 0.10        # 10% trailing stop from peak since entry
EXPOSURE = 0.97


class SP500SkipmonTrailstop(Strategy):
    """SP500 skip-month momentum with per-stock 10% trailing stop exits.

    Entry/refresh monthly; trailing stop monitored every bar.
    SPY 200d bear gate to IEF.
    """

    def __init__(
        self,
        entry_refresh: int = ENTRY_REFRESH,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        trend_window: int = TREND_WINDOW,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        trail_stop: float = TRAIL_STOP,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            entry_refresh=entry_refresh,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            trend_window=trend_window,
            top_k=top_k,
            vol_window=vol_window,
            trail_stop=trail_stop,
            exposure=exposure,
        )
        self.entry_refresh = int(entry_refresh)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.trend_window = int(trend_window)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.trail_stop = float(trail_stop)
        self.exposure = float(exposure)

        # Tracking state: {sym: peak_price_since_entry}
        self._peaks: dict[str, float] = {}
        # Current ranked portfolio (set at last refresh)
        self._current_targets: set[str] = set()
        # Inverse-vol weights at last refresh
        self._target_weights: dict[str, float] = {}

    def on_start(self) -> None:
        self._peaks = {}
        self._current_targets = set()
        self._target_weights = {}

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + self.vol_window + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        orders: list[Order] = []

        # --- Update peaks for open positions ---
        for sym, pos in ctx.positions.items():
            if pos.size > 0 and sym in live:
                px = live[sym]
                old_peak = self._peaks.get(sym, px)
                self._peaks[sym] = max(old_peak, px)

        # --- Check trailing stops on ALL current positions ---
        # Stop fires if price drops > trail_stop% from peak
        stopped_out: set[str] = set()
        for sym, pos in list(ctx.positions.items()):
            if pos.size > 0 and sym in live:
                px = live[sym]
                peak = self._peaks.get(sym, px)
                if peak > 0 and (px / peak - 1.0) < -self.trail_stop:
                    # Trailing stop triggered — exit immediately
                    orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))
                    stopped_out.add(sym)
                    self._peaks.pop(sym, None)
                    self._current_targets.discard(sym)

        # --- Full rebalance / entry refresh (monthly) ---
        is_rebalance = (ctx.idx % self.entry_refresh == 0)

        # --- SPY 200d SMA gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return orders
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window + 2:
            return orders
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        if not spy_bull:
            # Bear market: liquidate everything and go to IEF
            for sym, pos in list(ctx.positions.items()):
                if sym not in stopped_out and pos.size != 0:
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                    orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
            self._peaks.clear()
            self._current_targets.clear()
            self._target_weights.clear()
            if "IEF" in closes_now.index:
                ief_price = live.get("IEF")
                if ief_price and ief_price > 0:
                    tgt_shares = int(equity * self.exposure / ief_price)
                    cur = int(ctx.position("IEF").size)
                    delta = tgt_shares - cur
                    if abs(delta) >= 1:
                        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                        orders.append(Order(side=side, size=abs(delta), symbol="IEF"))
            return orders

        if is_rebalance:
            # Re-rank stocks and set new targets
            need = self.mom_lookback + self.mom_skip + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 1:
                return orders

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue
                # Skip-month momentum
                p_end = float(col.iloc[-self.mom_skip - 1])
                p_start = float(col.iloc[-(self.mom_lookback + self.mom_skip)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Inverse-vol weight
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < self.top_k:
                # Not enough candidates — IEF
                new_targets: dict[str, float] = {}
                if "IEF" in closes_now.index:
                    new_targets["IEF"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return orders
                new_targets = {sym: self.exposure * inv_vols[sym] / iv_sum for sym in ranked}

            self._current_targets = set(new_targets.keys())
            self._target_weights = new_targets

            # Liquidate positions not in new targets (and not already stopped out)
            for sym, pos in list(ctx.positions.items()):
                if sym not in stopped_out and sym not in new_targets and pos.size != 0:
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                    orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
                    self._peaks.pop(sym, None)

            # Size to new targets
            for sym, weight in new_targets.items():
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
                # Initialize peak for new buys
                if delta > 0 and sym not in self._peaks:
                    self._peaks[sym] = price

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["SPY", "IEF"]


NAME = "sp500_skipmon_trailstop"
HYPOTHESIS = (
    "SP500 top-15 skip-month momentum (126d-skip-21d) with per-stock 10% trailing stop exits: "
    "selection based on skip-month rank, inverse-vol weighted, SPY 200d bear gate to IEF; "
    "trailing stops replace fixed biweekly exit — adaptive name-level risk control orthogonal "
    "to vol-targeting mechanism"
)

UNIVERSE = _universe

STRATEGY = SP500SkipmonTrailstop()
