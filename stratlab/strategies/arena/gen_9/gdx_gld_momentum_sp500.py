"""GDX/GLD Momentum-Differential SP500 Gate — gen_9 sonnet-2 (variant 2)

Hypothesis: Instead of a z-score of the ratio level, use the simple return
differential between GDX and GLD as a timing signal. When GDX outperforms
GLD on a rolling 42-day return (miners beating metal), the commodity complex
signals risk-on. We use this as a secondary gate alongside the SPY 200d SMA.

Binary regime (simpler than 3-tier z-score):
  - When GDX 42d return > GLD 42d return (miners leading) AND SPY above 200d SMA:
    hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97%.
  - Otherwise: hold IEF 97% (intermediate bonds, preserves capital).

The simpler binary decision avoids the "neutral" bucket that diluted performance
in the 3-tier z-score variant. The IEF defensive (not TLT) reduces duration
sensitivity in the defensive mode.

Differentiation from leaderboard:
  - No prior round uses the GDX/GLD return differential as the regime gate.
  - gen7_sp500_126d_stock_50sma_goldencross uses SPY 50d vs 150d golden cross.
  - gen8_opus1_longend_slope_equity_gate uses TYX-TNX yield curve slope.
  - This uses commodity-equity (GDX) vs commodity-metal (GLD) return spread —
    a fundamentally different cross-asset linkage.

GDX inception: 2006. GLD inception: 2004. Both cover IS fully.
IS window: 2010-2018, biweekly rebalance (every 10 bars).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month momentum for stock ranking
TREND_WINDOW = 200        # SPY/stock 200d SMA
SIGNAL_WINDOW = 42        # GDX vs GLD return comparison window
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["GDX", "GLD", "IEF", "SPY"]


UNIVERSE = _universe


class GdxGldMomentumSp500(Strategy):
    """SP500 63d momentum gated by GDX-vs-GLD 42d return differential."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, SIGNAL_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA market gate ---
        spy_hist = ctx.history("SPY")
        spy_bull = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bull = spy_price > 0 and spy_price > spy_sma

        if not spy_bull:
            target = {"IEF": EXPOSURE}
        else:
            # --- GDX vs GLD 42d return comparison ---
            gdx_hist = ctx.history("GDX")
            gld_hist = ctx.history("GLD")

            gdx_risk_on = False
            if len(gdx_hist) >= SIGNAL_WINDOW + 1 and len(gld_hist) >= SIGNAL_WINDOW + 1:
                gdx_close = gdx_hist["close"]
                gld_close = gld_hist["close"]

                gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-SIGNAL_WINDOW] - 1.0)
                gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-SIGNAL_WINDOW] - 1.0)

                if np.isfinite(gdx_ret) and np.isfinite(gld_ret):
                    gdx_risk_on = gdx_ret > gld_ret

            if not gdx_risk_on:
                target = {"IEF": EXPOSURE}
            else:
                # GDX outperforming GLD + SPY bull: enter SP500 momentum
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"IEF": EXPOSURE}
                else:
                    live = {s: float(closes[s]) for s in closes.index
                            if closes[s] > 0 and s not in ("GDX", "GLD", "IEF", "SPY")}

                    scores: dict[str, float] = {}
                    for sym in live:
                        if sym not in prices_window.columns:
                            continue
                        col = prices_window[sym].dropna()
                        if len(col) < MOMENTUM_WINDOW:
                            continue
                        p_end = float(col.iloc[-1])
                        p_start = float(col.iloc[-MOMENTUM_WINDOW])
                        if p_start <= 0:
                            continue
                        r = p_end / p_start - 1.0
                        if np.isfinite(r):
                            scores[sym] = r

                    if len(scores) < TOP_K:
                        target = {"IEF": EXPOSURE}
                    else:
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                        # 200d SMA stock-level filter
                        selected = []
                        for sym, _ in ranked:
                            if len(selected) >= TOP_K:
                                break
                            hist = ctx.history(sym)
                            if len(hist) < TREND_WINDOW:
                                continue
                            sma = float(hist["close"].iloc[-TREND_WINDOW:].mean())
                            price = live.get(sym, 0.0)
                            if price > sma:
                                selected.append(sym)

                        if not selected:
                            target = {"IEF": EXPOSURE}
                        else:
                            target = {sym: EXPOSURE / len(selected) for sym in selected}

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live_all.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live_all.get(sym, 0.0)
            if price <= 0:
                continue
            tgt_shares = int(equity * weight / price)
            current = ctx.position(sym).size
            delta = tgt_shares - current
            if delta == 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


NAME = "gdx_gld_momentum_sp500"
HYPOTHESIS = (
    "GDX vs GLD 42d return differential as commodity-risk gate for SP500 momentum: "
    "when GDX outperforms GLD on 42d return (miners beating metal) AND SPY above 200d SMA, "
    "hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97%; "
    "otherwise hold IEF 97%; biweekly rebalance."
)

STRATEGY = GdxGldMomentumSp500()
