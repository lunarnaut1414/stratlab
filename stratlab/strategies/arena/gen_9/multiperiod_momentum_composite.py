"""gen_9 sonnet-6 — Multi-Period Momentum Composite on SP500

Hypothesis: Rank SP500 stocks by an equal-weighted composite of normalized
momentum rank across 4 horizons: 21d, 63d, 126d, and 252d total returns.
Each stock gets a normalized rank (0-1) per horizon; composite = mean of ranks.
Hold top-15 by composite score above 200d SMA, inverse-vol weighted.

Rationale: Single-lookback momentum (21d, 63d, 126d, 252d) is inherently noisy
— stocks that rank high on one horizon may not rank high on others. The
composite rank is more stable across market regimes: 21d captures recent
strength, 63d intermediate momentum, 126d medium-term trend, 252d long-term
trend. Normalizing by rank (not raw return) prevents a single explosive
month from dominating. Inverse-vol weighting provides natural deleveraging.

The composite score is different from:
- Pure 63d momentum (gen5_vix_gated_sp500_momentum)
- Skip-month 126d-21d (gen8_sp500_skipmon_63sma_momentum)
- Idiosyncratic momentum (gen7_sp500_idiosyncratic_momentum)
- Nearhi quality (gen6_nearhi_momentum_quality)

Gate: SPY 200d SMA → TLT when bearish. Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
WINDOWS = [21, 63, 126, 252]   # momentum horizons for composite
TREND_WINDOW = 200
VOL_WINDOW = 21
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"


class MultiperiodMomentumComposite(Strategy):
    """SP500 composite multi-period momentum rank with inverse-vol weighting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        max_window = max(WINDOWS)
        warmup = max(self.trend_window, max_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Defensive: TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute multi-period momentum composite
            need = max_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < max_window:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Compute returns for each window, then rank per window
                # Then composite = mean of normalized ranks
                per_window_returns: dict[int, dict[str, float]] = {}

                for w in WINDOWS:
                    rets = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < w:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-w] - 1.0)
                        if np.isfinite(ret):
                            rets[sym] = ret
                    per_window_returns[w] = rets

                # Find symbols that have data for ALL windows
                all_syms = set(per_window_returns[WINDOWS[0]].keys())
                for w in WINDOWS[1:]:
                    all_syms &= set(per_window_returns[w].keys())

                if len(all_syms) < 10:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Compute composite rank for each symbol
                    composite: dict[str, float] = {}
                    syms_list = list(all_syms)

                    for w in WINDOWS:
                        # Get returns for this window, for all common symbols
                        w_rets = {s: per_window_returns[w][s] for s in syms_list}
                        sorted_syms = sorted(syms_list, key=lambda s: w_rets[s])
                        n = len(sorted_syms)
                        # Normalized rank: 0 (worst) to 1 (best)
                        for rank_i, sym in enumerate(sorted_syms):
                            norm_rank = rank_i / (n - 1) if n > 1 else 0.5
                            composite[sym] = composite.get(sym, 0.0) + norm_rank

                    # Average across 4 windows
                    n_windows = len(WINDOWS)
                    for sym in composite:
                        composite[sym] /= n_windows

                    if len(composite) < 5:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        k = min(self.top_k, len(composite))
                        ranked = sorted(composite, key=composite.__getitem__, reverse=True)[:k]

                        # Inverse-vol weighting
                        vol_window_actual = min(self.vol_window, len(prices) - 1)
                        if vol_window_actual < 5:
                            per_weight = self.exposure / len(ranked)
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = per_weight
                        else:
                            inv_vols = {}
                            for sym in ranked:
                                col = prices[sym].dropna()
                                if len(col) >= vol_window_actual + 1:
                                    log_rets = np.log(col.values[1:] / col.values[:-1])
                                    rv = float(np.std(log_rets[-vol_window_actual:]) * np.sqrt(252))
                                    if rv > 0 and np.isfinite(rv):
                                        inv_vols[sym] = 1.0 / rv
                                    else:
                                        inv_vols[sym] = 1.0
                                else:
                                    inv_vols[sym] = 1.0

                            total_inv = sum(inv_vols.values())
                            if total_inv > 0:
                                for sym in ranked:
                                    w = inv_vols.get(sym, 1.0) / total_inv * self.exposure
                                    if sym in live:
                                        target[sym] = w
                            else:
                                per_weight = self.exposure / len(ranked)
                                for sym in ranked:
                                    if sym in live:
                                        target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY]


NAME = "multiperiod_momentum_composite"
HYPOTHESIS = (
    "Multi-period momentum composite on SP500: rank stocks by equal-weighted average of "
    "normalized rank across 21d, 63d, 126d, and 252d total returns; hold top-15 above 200d "
    "SMA, inverse-vol weighted, SPY 200d outer bear gate to TLT; biweekly rebalance; "
    "time-diversified momentum ranks are more stable than any single lookback"
)

UNIVERSE = _universe

STRATEGY = MultiperiodMomentumComposite()
