"""Popular ETF absolute+relative momentum — dual momentum filter.

Hypothesis (sonnet-10, gen_10):
    Rank all popular_etfs by 63d return (relative momentum). Hold the top-5
    ETFs that ALSO have positive absolute 63d return. Weight inversely by 21d
    realized vol. IEF when fewer than 3 ETFs qualify. Biweekly rebalance.
    Add SPY 200d SMA as outer gate: when SPY is below 200d SMA, hold IEF 97%.

Rationale:
  - The dual absolute+relative filter ensures we only hold ETFs in actual
    uptrends (not just the least-bad performers in a bear).
  - The popular_etfs universe includes sector, international, commodity, bond
    and thematic ETFs. Unlike SP500 stock selection strategies, the PORTFOLIO
    COMPOSITION itself changes regime (may hold TLT, GLD, XLU during risk-off
    while still beating the absolute 63d return threshold).
  - Inverse-vol weighting further diversifies risk across different asset types.
  - Biweekly gives 100+ trades easily in this universe.

Diversification angle vs leaderboard:
  - gen5 dead-end attempted popular_etf abs momentum but WITHOUT the relative
    rank (top-5 qualifier) OR the SPY 200d gate. Both additions improve Calmar.
  - gen8_rsp_spy_breadth_qqq_rotation, gen8_iwm_spy_size_regime_qqq_rotation:
    those route between specific ETF pairs based on relative strength. This
    strategy ranks ALL popular ETFs and selects diversified top-5 cross-
    sectionally — different mechanism and much larger universe.
  - Popular_etfs universe includes bonds/commodities/sectors that move
    differently from SP500 stocks in loss modes — low loss-mode corr expected.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
MOM_WINDOW = 63            # 63d momentum window
VOL_WINDOW = 21            # 21d realized vol for inverse-vol weighting
SPY_TREND_WINDOW = 200     # outer bear gate
TOP_K = 5                  # top ETFs to hold
MIN_QUALIFY = 3            # minimum qualifying ETFs before IEF fallback
EXPOSURE = 0.97


class PopularETFAbsRelMomentum(Strategy):
    """Popular ETF dual absolute+relative momentum; inv-vol weighted; SPY 200d gate."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(MOM_WINDOW, SPY_TREND_WINDOW) + VOL_WINDOW + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW + 5:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            if "IEF" in closes_now.index:
                target["IEF"] = EXPOSURE
        else:
            need = MOM_WINDOW + VOL_WINDOW + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOM_WINDOW:
                return []

            scores: dict[str, float] = {}
            inv_vols: dict[str, float] = {}

            for sym in prices.columns:
                col = prices[sym].dropna()
                if len(col) < MOM_WINDOW + 2:
                    continue

                # 63d return (relative ranking criterion)
                p_end = float(col.iloc[-1])
                p_start = float(col.iloc[-MOM_WINDOW])
                if p_start <= 0 or not np.isfinite(p_start):
                    continue
                ret = p_end / p_start - 1.0
                if not np.isfinite(ret):
                    continue

                # Absolute momentum filter: 63d return must be positive
                if ret <= 0:
                    continue

                # Per-symbol inverse-vol weight
                tail = col.values[-(VOL_WINDOW + 1):]
                if len(tail) < VOL_WINDOW + 1:
                    continue
                logr = np.log(tail[1:] / tail[:-1])
                rv = float(np.std(logr))
                if rv <= 1e-6 or not np.isfinite(rv):
                    continue

                scores[sym] = ret
                inv_vols[sym] = 1.0 / rv

            # Filter: need at least MIN_QUALIFY candidates
            if len(scores) < MIN_QUALIFY:
                if "IEF" in closes_now.index:
                    target["IEF"] = EXPOSURE
            else:
                k = min(TOP_K, len(scores))
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                iv_sum = sum(inv_vols[s] for s in ranked)
                if iv_sum <= 0:
                    return []
                for sym in ranked:
                    target[sym] = EXPOSURE * inv_vols[sym] / iv_sum

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


UNIVERSE = "popular_etfs"

NAME = "popular_etf_abs_rel_momentum"
HYPOTHESIS = (
    "Popular ETF cross-sectional momentum: rank all popular_etfs by 63d return, hold top-5 "
    "with positive absolute 63d return (dual absolute+relative momentum filter); inverse-vol "
    "weighted (21d vol); when fewer than 3 ETFs qualify hold IEF; biweekly rebalance; "
    "SPY 200d outer gate to IEF — portfolio naturally includes bonds/gold/commodity ETFs "
    "during risk-off, creating genuinely different return timing than SP500 stock momentum"
)

STRATEGY = PopularETFAbsRelMomentum()
