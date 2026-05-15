"""GDX/GLD Miner-vs-Metal Z-Score SP500 Gate — gen_9 sonnet-2

Hypothesis: The GDX/GLD price ratio captures the relative performance of gold
miners vs physical gold. When miners lead gold (high z-score), it signals
risk-on sentiment within the commodity complex — miners are leveraged to gold
prices and outperform in strong growth + inflation environments. When gold
leads miners (low z-score), it signals flight-to-safety, often accompanied by
earnings stress in mining companies.

Three-tier regime:
  - High z-score (>+0.5): GDX outperforming GLD → commodity-risk-on →
    hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97% exposure.
  - Low z-score (<-0.5): GLD outperforming GDX → flight-to-safety →
    hold TLT 97% (full defensive).
  - Neutral (-0.5 to +0.5): ambiguous → hold SPY 60%+IEF 37%.

SPY 200d SMA outer bear gate: if SPY below 200d SMA, force TLT regardless.

Differentiation from leaderboard:
  - No prior round uses GDX/GLD as the primary regime signal.
  - gen7_sp500_idiosyncratic_momentum uses beta-adjusted residual alpha.
  - gen8_sp500_credit_zscore_3tier uses JNK/LQD credit z-score — completely
    different signal domain (commodity complex vs credit markets).
  - The miner-vs-metal signal captures a different cross-asset stress channel:
    when gold miners underperform gold, it often leads equity weakness by
    several weeks (miners are leveraged beta to growth expectations).

IS window: 2010-2018 | GDX and GLD both have full data from 2004.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month momentum for stock ranking
TREND_WINDOW = 200        # SPY 200d SMA bear gate
ZSCORE_WINDOW = 90        # Rolling window for GDX/GLD ratio z-score
Z_HIGH = 0.0              # Above: risk-on (SP500 momentum) — default is risk-on
Z_LOW = -1.0              # Below: risk-off (TLT) — only deep divergence triggers defensive
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    return sp500_tickers() + ["GDX", "GLD", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class GdxGldMinerMetalZscore(Strategy):
    """SP500 63d momentum with GDX/GLD ratio 90d z-score 3-tier regime gate."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, ZSCORE_WINDOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live_all = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live_all.get("SPY", 0.0)
            spy_bear = spy_price > 0 and spy_price <= spy_sma

        if spy_bear:
            target = {"TLT": EXPOSURE}
        else:
            # --- Compute GDX/GLD ratio z-score ---
            gdx_hist = ctx.history("GDX")
            gld_hist = ctx.history("GLD")

            if len(gdx_hist) < ZSCORE_WINDOW + 5 or len(gld_hist) < ZSCORE_WINDOW + 5:
                return []

            gdx_close = gdx_hist["close"].tail(ZSCORE_WINDOW + 5)
            gld_close = gld_hist["close"].tail(ZSCORE_WINDOW + 5)

            min_len = min(len(gdx_close), len(gld_close))
            if min_len < ZSCORE_WINDOW:
                return []

            gdx_vals = gdx_close.values[-min_len:]
            gld_vals = gld_close.values[-min_len:]

            # GDX/GLD ratio (miners per unit of gold)
            gld_safe = np.where(gld_vals > 0, gld_vals, np.nan)
            ratio = gdx_vals / gld_safe

            # Use last ZSCORE_WINDOW values
            ratio_window = ratio[-ZSCORE_WINDOW:]
            valid = ratio_window[~np.isnan(ratio_window)]
            if len(valid) < 20:
                return []

            ratio_mean = float(np.mean(valid))
            ratio_std = float(np.std(valid))
            if ratio_std <= 0 or not np.isfinite(ratio_std):
                return []

            current_ratio = valid[-1]
            z_score = (current_ratio - ratio_mean) / ratio_std

            # --- Route based on z-score ---
            if z_score < Z_LOW:
                # GLD outperforming GDX: flight-to-safety → TLT defensive
                target = {"TLT": EXPOSURE}
            elif z_score <= Z_HIGH:
                # Neutral: SPY + IEF blend
                target = {"SPY": 0.60, "IEF": 0.37}
            else:
                # GDX outperforming GLD: commodity risk-on → SP500 momentum
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"SPY": 0.60, "IEF": 0.37}
                else:
                    live = {s: float(closes[s]) for s in closes.index
                            if closes[s] > 0 and s not in ("GDX", "GLD", "TLT", "IEF", "SPY")}

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
                        target = {"SPY": 0.60, "IEF": 0.37}
                    else:
                        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                        # Apply 200d SMA stock-level filter
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
                            target = {"SPY": 0.60, "IEF": 0.37}
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


NAME = "gdx_gld_miner_metal_zscore"
HYPOTHESIS = (
    "GDX/GLD miner-vs-metal ratio 90d z-score as commodity-risk signal: "
    "z>+0.5 (miners lead gold, commodity risk-on) hold top-15 SP500 stocks 63d momentum 97%; "
    "z<-0.5 (gold leads miners, flight-to-safety) hold TLT 97%; "
    "neutral: hold SPY 60%+IEF 37%; SPY 200d outer bear gate; biweekly rebalance."
)

STRATEGY = GdxGldMinerMetalZscore()
