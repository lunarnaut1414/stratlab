"""SP500 126d-skip-21d momentum with per-stock trailing stop + portfolio vol-targeting.

Hypothesis (sonnet-3, gen_10):
    Combine three proven mechanisms from gen_9 into one strategy:
      1. 126d-skip-21d (Jegadeesh-Titman) momentum ranking — from
         gen9_sp500_voltarget_skipmon (OOS 0.86, 80% retention)
      2. Per-stock 10% trailing stop exits (from gen7_opus1_sp500_idio_trailstop
         OOS 0.37, but the exit mechanism itself is valid)
      3. Portfolio 14% vol-targeting (from gen9 voltarget family)

    Combination rationale:
      - Skip-month ranking avoids buying stocks about to mean-revert (short-term
        reversal) by excluding the most recent month from the lookback.
      - Trailing stop exits let winners run longer while cutting individual losers
        early — avoids the fixed biweekly wholesale replacement that may exit
        winners prematurely or hold losers too long.
      - Portfolio vol-targeting provides aggregate risk control independent of
        VIX level or any macro signal — structurally regime-invariant.

    The gen_7 trailing-stop strategy used idiosyncratic momentum (beta-adjusted)
    without vol-targeting. Gen_9 voltarget used skip-month without trailing stops.
    This strategy combines skip-month ENTRY selection with trailing-stop EXITS
    and vol-target SIZING — none of the existing strategies use all three.

    Design:
      - Monthly (21-bar) refresh: recompute top-K by 126d-skip-21d momentum.
        Open slots (from exits or new round) fill with top-K names.
      - Daily check: for each holding, if current price drops more than 10%
        below the highest price since entry, exit that position.
      - Portfolio vol-target: on each monthly refresh, scale total exposure
        to hit 14% annualized portfolio vol (30d realized), clipped 50-97%.
      - SPY 200d SMA outer gate: IEF defensive when bear.
      - Inverse-vol cross-sectional weighting for position sizes.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

ENTRY_REFRESH = 21          # monthly rebalance / new entry check
MOM_LOOKBACK = 126          # 6-month momentum
MOM_SKIP = 21               # skip last month (Jegadeesh-Titman)
TRAIL_STOP = 0.10           # 10% trailing stop from peak
VOL_WINDOW = 21             # inverse-vol lookback
SPY_TREND_WINDOW = 200      # outer gate
TOP_K = 15
EXPOSURE_MIN = 0.50
EXPOSURE_MAX = 0.97
VOL_TARGET = 0.14           # 14% portfolio vol target
PORT_VOL_WINDOW = 30        # realized vol lookback
ANNUALIZATION = 252


class Sp500SkipmonTrailstopVoltarget(Strategy):
    """SP500 126d-skip-21d momentum with per-stock 10% trailing stop and 14%
    portfolio vol-targeting. Monthly entry refresh; SPY 200d SMA gate to IEF.
    """

    def __init__(
        self,
        entry_refresh: int = ENTRY_REFRESH,
        mom_lookback: int = MOM_LOOKBACK,
        mom_skip: int = MOM_SKIP,
        trail_stop: float = TRAIL_STOP,
        vol_window: int = VOL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure_min: float = EXPOSURE_MIN,
        exposure_max: float = EXPOSURE_MAX,
        vol_target: float = VOL_TARGET,
        port_vol_window: int = PORT_VOL_WINDOW,
    ) -> None:
        super().__init__(
            entry_refresh=entry_refresh,
            mom_lookback=mom_lookback,
            mom_skip=mom_skip,
            trail_stop=trail_stop,
            vol_window=vol_window,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure_min=exposure_min,
            exposure_max=exposure_max,
            vol_target=vol_target,
            port_vol_window=port_vol_window,
        )
        self.entry_refresh = int(entry_refresh)
        self.mom_lookback = int(mom_lookback)
        self.mom_skip = int(mom_skip)
        self.trail_stop = float(trail_stop)
        self.vol_window = int(vol_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure_min = float(exposure_min)
        self.exposure_max = float(exposure_max)
        self.vol_target = float(vol_target)
        self.port_vol_window = int(port_vol_window)

        # State: track entry_price and peak_since_entry per holding
        self._peak: dict[str, float] = {}
        self._entry: dict[str, float] = {}
        self._exposure: float = EXPOSURE_MAX  # last computed exposure

    def on_start(self) -> None:
        self._peak = {}
        self._entry = {}
        self._exposure = self.exposure_max

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.mom_lookback + self.mom_skip + self.port_vol_window + 10
        if ctx.idx < warmup:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}

        orders: list[Order] = []

        # --- SPY 200d SMA gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window + 2:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma

        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        if not spy_bull:
            # Bear market: exit all equity, go to IEF
            self._peak.clear()
            self._entry.clear()
            target_sym = "IEF"
            for sym, pos in list(ctx.positions.items()):
                if sym != target_sym and pos.size != 0:
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                    orders.append(Order(side=side, size=abs(pos.size), symbol=sym))
            # Size IEF
            price_ief = live.get(target_sym)
            if price_ief and price_ief > 0:
                tgt_shares = int(equity * self.exposure_max / price_ief)
                cur = int(ctx.position(target_sym).size)
                delta = tgt_shares - cur
                if abs(delta) >= 1:
                    orders.append(Order(
                        side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                        size=abs(delta),
                        symbol=target_sym,
                    ))
            return orders

        # --- Daily trailing-stop check on existing positions ---
        for sym in list(self._peak.keys()):
            price = live.get(sym)
            if price is None or price <= 0:
                continue
            # Update peak
            if price > self._peak[sym]:
                self._peak[sym] = price
            # Check trailing stop: exit if price dropped more than trail_stop from peak
            if self._peak[sym] > 0 and (price / self._peak[sym]) < (1.0 - self.trail_stop):
                # Trailing stop hit — exit this position
                pos = ctx.position(sym)
                if pos.size != 0:
                    side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                    orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
                del self._peak[sym]
                self._entry.pop(sym, None)

        # --- Monthly entry refresh (recompute top-K) ---
        if ctx.idx % self.entry_refresh == 0:
            need = self.mom_lookback + self.mom_skip + 2
            prices = ctx.closes_window(need)
            if len(prices) < need - 2:
                return orders

            # Compute skip-month scores and inv-vol weights
            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}
            for sym in prices.columns:
                if sym in ("SPY", "IEF"):
                    continue
                col = prices[sym].dropna()
                if len(col) < self.mom_lookback + self.mom_skip:
                    continue
                # Skip-month: p_end at -mom_skip, p_start at -(mom_lookback + mom_skip)
                p_end = float(col.iloc[-self.mom_skip - 1])
                p_start = float(col.iloc[-(self.mom_lookback + self.mom_skip)])
                if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue
                # Inverse-vol
                tail = col.values[-(self.vol_window + 1):]
                if len(tail) < self.vol_window + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue
                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            if len(scores) < 5:
                return orders

            k = min(self.top_k, len(scores))
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]

            # --- Portfolio vol-targeting ---
            port_prices = ctx.closes_window(self.port_vol_window + 5)
            port_rets = []
            n_rows = len(port_prices)
            for row_idx in range(1, n_rows):
                row_ret = 0.0
                count = 0
                for sym in ranked:
                    if sym not in port_prices.columns:
                        continue
                    p_now = port_prices[sym].iloc[row_idx]
                    p_prev = port_prices[sym].iloc[row_idx - 1]
                    if np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0:
                        row_ret += np.log(float(p_now) / float(p_prev))
                        count += 1
                if count > 0:
                    port_rets.append(row_ret / count)

            if len(port_rets) >= 10:
                daily_vol = float(np.std(port_rets))
                annual_vol = daily_vol * np.sqrt(ANNUALIZATION)
                scale = self.vol_target / annual_vol if annual_vol > 1e-6 else 1.0
                self._exposure = float(np.clip(scale, self.exposure_min, self.exposure_max))
            else:
                self._exposure = self.exposure_max

            iv_sum = sum(inv_vols[s] for s in ranked)
            if iv_sum <= 0:
                return orders

            # Build target weights
            target: dict[str, float] = {}
            for sym in ranked:
                target[sym] = self._exposure * inv_vols[sym] / iv_sum

            # Exit positions not in target (no longer top-K)
            active_now = set(k for k, p in ctx.positions.items() if p.size != 0)
            for sym in active_now:
                if sym not in target and sym != "IEF":
                    pos = ctx.position(sym)
                    if pos.size != 0:
                        side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                        orders.append(Order(side=side, size=abs(int(pos.size)), symbol=sym))
                    self._peak.pop(sym, None)
                    self._entry.pop(sym, None)

            # Also exit IEF if we're in bull and have IEF
            ief_pos = ctx.position("IEF")
            if ief_pos.size != 0:
                side = OrderSide.SELL if ief_pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(int(ief_pos.size)), symbol="IEF"))

            # Enter / resize target positions
            for sym, weight in target.items():
                price = live.get(sym)
                if not price or price <= 0:
                    continue
                tgt_shares = int(equity * weight / price)
                cur = int(ctx.position(sym).size)
                delta = tgt_shares - cur
                if abs(delta) < 1:
                    continue
                orders.append(Order(
                    side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    size=abs(delta),
                    symbol=sym,
                ))
                # Initialize/update peak tracking
                if sym not in self._peak or delta > 0:
                    self._peak[sym] = price
                    self._entry[sym] = price

        return orders


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["IEF", "SPY"]


UNIVERSE = _universe

NAME = "sp500_skipmon_trailstop_voltarget"
HYPOTHESIS = (
    "SP500 top-15 by 126d-skip-21d momentum with per-stock 10pct trailing stop from 21d peak "
    "combined with portfolio 14pct vol-target sizing; SPY 200d SMA gate; IEF defensive; "
    "entry refresh every 21 bars for new top-15 selection; trailing stop triggers per-stock "
    "exits between rebalances — combines gen_9 best skip-month mechanism with adaptive "
    "per-stock exit rule"
)

STRATEGY = Sp500SkipmonTrailstopVoltarget()
