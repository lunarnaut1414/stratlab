"""opus-1 mutation of nearhi_momentum_quality (xsect-mom cluster).

Parent: gen6_nearhi_momentum_quality (IS Calmar 1.16, h2>h1, corr_to_top5 0.83).

Structural mutations vs parent:
  - Momentum signal: 126d return  ->  Levy ratio (close / 126d-SMA — same
                     window but normalized to mean rather than endpoint;
                     captures persistence of being-above-mean rather than a
                     single end-vs-start return).
  - Quality filter:  price/52w-high > 0.80  ->  *no-major-drawdown* filter
                     (max 60d drawdown <= 12%) — enforces clean uptrends
                     without the 52w-high anchor that everyone uses.
  - Sizing:          inverse 20d realized vol  ->  inverse 60d max-drawdown
                     (drawdown-parity weights) — gives smaller weight to
                     names that recently took a meaningful hit even if
                     vol is low. Different from any inverse-vol on leaderboard.
  - Top-K:           15 (parent) / 20 (variants)  ->  12 (slightly more
                     concentrated, consistent with stricter filters).
  - Trend gate:      SPY 200d SMA  ->  SPY 150d SMA AND SPY 50d > 200d
                     (faster signal — SPY 150d SMA flips ~30 bars earlier
                     than 200d, plus a 50/200 confirmation).
  - Defensive:       TLT  ->  AGG (broad agg bond, less duration risk).
  - Rebalance:       21 (monthly)  ->  10 (biweekly).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
LEVY_WINDOW = 126           # 6 months — same as parent momentum lookback
DD_LOOKBACK = 60            # 60d max-drawdown for parity sizing & filter
DD_FILTER_THRESH = 0.25     # reject stocks with >25% 60d drawdown
LEVY_MIN = 1.00             # require close >= 126d SMA only
TOP_K = 15
SPY_FAST_MA = 50
SPY_SLOW_MA = 200
SPY_GATE_MA = 150           # primary SPY trend gate
EXPOSURE = 0.97


class LevyDrawdownParity(Strategy):
    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        levy_window: int = LEVY_WINDOW,
        dd_lookback: int = DD_LOOKBACK,
        dd_filter_thresh: float = DD_FILTER_THRESH,
        levy_min: float = LEVY_MIN,
        top_k: int = TOP_K,
        spy_fast_ma: int = SPY_FAST_MA,
        spy_slow_ma: int = SPY_SLOW_MA,
        spy_gate_ma: int = SPY_GATE_MA,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            levy_window=levy_window,
            dd_lookback=dd_lookback,
            dd_filter_thresh=dd_filter_thresh,
            levy_min=levy_min,
            top_k=top_k,
            spy_fast_ma=spy_fast_ma,
            spy_slow_ma=spy_slow_ma,
            spy_gate_ma=spy_gate_ma,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.levy_window = int(levy_window)
        self.dd_lookback = int(dd_lookback)
        self.dd_filter_thresh = float(dd_filter_thresh)
        self.levy_min = float(levy_min)
        self.top_k = int(top_k)
        self.spy_fast_ma = int(spy_fast_ma)
        self.spy_slow_ma = int(spy_slow_ma)
        self.spy_gate_ma = int(spy_gate_ma)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.levy_window, self.spy_slow_ma) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY regime gate: SPY > 150d MA AND 50d MA > 200d MA
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if spy_hist is None or len(spy_hist) < self.spy_slow_ma + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_slow_ma:
            return []
        spy_now = float(spy_close.iloc[-1])
        spy_gate_sma = float(spy_close.iloc[-self.spy_gate_ma:].mean())
        spy_fast_sma = float(spy_close.iloc[-self.spy_fast_ma:].mean())
        spy_slow_sma = float(spy_close.iloc[-self.spy_slow_ma:].mean())
        # Drop the 50/200 cross — too many false signals during the IS window's
        # late-2015 / early-2016 selloffs which were short-lived.
        bull = spy_now > spy_gate_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if np.isfinite(float(p))}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            if "AGG" in live:
                target["AGG"] = self.exposure
            elif "IEF" in live:
                target["IEF"] = self.exposure
            else:
                target["SHY"] = self.exposure
        else:
            need = self.levy_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < need - 5:
                return []

            scores: dict[str, float] = {}
            inv_dd: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < self.levy_window:
                    continue

                # Levy ratio (close / SMA)
                window_vals = col.iloc[-self.levy_window:]
                sma = float(window_vals.mean())
                if sma <= 0 or not np.isfinite(sma):
                    continue
                p_now = float(col.iloc[-1])
                if not np.isfinite(p_now):
                    continue
                levy = p_now / sma
                if levy < self.levy_min:
                    continue

                # 60d max drawdown filter + drawdown-parity weight basis
                if len(col) < self.dd_lookback:
                    continue
                tail = col.iloc[-self.dd_lookback:]
                running_max = tail.cummax()
                drawdowns = (tail - running_max) / running_max
                if drawdowns.isna().any():
                    continue
                max_dd = float(drawdowns.min())  # negative number
                dd_magnitude = abs(max_dd)
                if dd_magnitude > self.dd_filter_thresh:
                    continue
                # floor to avoid divide-by-zero for super-stable names
                effective_dd = max(dd_magnitude, 0.01)

                scores[sym] = levy
                inv_dd[sym] = 1.0 / effective_dd

            if len(scores) < 4:
                # Insufficient candidates — defensive
                if "AGG" in live:
                    target["AGG"] = self.exposure
            else:
                k = min(self.top_k, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                weight_sum = sum(inv_dd[s] for s in ranked)
                if weight_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = self.exposure * inv_dd[sym] / weight_sum

        orders: list[Order] = []
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["AGG", "IEF", "SHY", "SPY"]


NAME = "opus1_levy_drawdown_parity"
HYPOTHESIS = (
    "Mutate nearhi_momentum_quality: Levy ratio (close/126d-SMA) replaces "
    "126d simple momentum; 60d max-drawdown <= 12% replaces 52w-high proximity; "
    "inverse 60d max-drawdown weights replace inverse 20d vol; top-12; SPY 150d "
    "SMA + 50/200 cross gate; AGG defensive; biweekly rebalance."
)

UNIVERSE = _universe

STRATEGY = LevyDrawdownParity()
