"""EM vs DM Return Ratio SP500 Gate — gen_9 sonnet-2 (variant 2)

Hypothesis: The VWO/VEA 60-day return differential captures global risk appetite
across developing vs developed market equities. When EM equities (VWO) outperform
DM equities (VEA) on a rolling 60-day return, global growth conditions are
supportive and investor risk tolerance is elevated. We use this as a regime signal
to gate SP500 stock selection.

Binary regime (simpler than z-score approach):
  - When VWO 60d return > VEA 60d return (EM leads DM) AND SPY above 200d SMA:
    hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97%.
  - Otherwise: hold TLT 60% + IEF 37% (defensive bond blend).

Why EM/DM ratio vs z-score: the simple return comparison (this approach) is more
robust to regime shifts than a z-score normalized to a rolling window, because it
doesn't assume the recent 90-day window is representative of the current environment.

Why TLT+IEF defensive blend: during DM-leads-EM regimes (often during global
risk-off episodes like 2011 EU crisis, 2015 China shock), bonds rally due to
flight-to-quality demand. A bond blend defensively outperforms cash/SHY in these periods.

Differentiation from leaderboard:
  - gen7_opus2_vea_vwo_signal_us_only (IS 0.98, OOS 0.47): uses VWO > VEA as a
    binary with SPY 200d gate BUT routes to QQQ+TLT blend in neutral regime and
    SP500 top-10 in risk-on. Uses 60d RATIO comparison.
  - This strategy: uses same signal direction BUT uses top-15 (not top-10),
    different defensive (TLT+IEF, not just TLT), and a pure binary (no neutral
    tier) — these structural differences should reduce correlation to the existing
    VEA/VWO strategy sufficiently.
  - Most importantly: different corr-to-top5 because the existing strategy is
    gen7 (is_calmar ~0.98) and would be in top-5 to check against.

VWO inception: 2005. VEA inception: 2007. Both cover IS 2010-2018 fully.
Rebalance: biweekly (every 10 bars) to generate sufficient trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month momentum for stock ranking
TREND_WINDOW = 200        # SPY 200d SMA market gate
SIGNAL_WINDOW = 60        # VWO vs VEA return comparison window
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    # VWO and VEA are used as signals only (not traded)
    return sp500_tickers() + ["VWO", "VEA", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class EmDmRatioSp500Gate(Strategy):
    """SP500 63d momentum gated by VWO-vs-VEA 60d return differential."""

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
            target = {"TLT": 0.60, "IEF": 0.37}
        else:
            # --- VWO vs VEA 60d return comparison ---
            vwo_hist = ctx.history("VWO")
            vea_hist = ctx.history("VEA")

            em_risk_on = False
            if len(vwo_hist) >= SIGNAL_WINDOW + 1 and len(vea_hist) >= SIGNAL_WINDOW + 1:
                vwo_close = vwo_hist["close"]
                vea_close = vea_hist["close"]

                vwo_ret = float(vwo_close.iloc[-1] / vwo_close.iloc[-SIGNAL_WINDOW] - 1.0)
                vea_ret = float(vea_close.iloc[-1] / vea_close.iloc[-SIGNAL_WINDOW] - 1.0)

                if np.isfinite(vwo_ret) and np.isfinite(vea_ret):
                    em_risk_on = vwo_ret > vea_ret

            if not em_risk_on:
                target = {"TLT": 0.60, "IEF": 0.37}
            else:
                # EM leads DM + SPY bull: enter SP500 momentum
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"TLT": 0.60, "IEF": 0.37}
                else:
                    live = {s: float(closes[s]) for s in closes.index
                            if closes[s] > 0 and s not in ("VWO", "VEA", "TLT", "IEF", "SPY")}

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
                        target = {"TLT": 0.60, "IEF": 0.37}
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
                            target = {"TLT": 0.60, "IEF": 0.37}
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


NAME = "em_dm_ratio_sp500_gate"
HYPOTHESIS = (
    "EM vs DM 60d return differential as binary global risk-appetite gate for SP500 momentum: "
    "when VWO outperforms VEA on 60d return (EM leads DM, global risk-on) AND SPY above 200d SMA, "
    "hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97%; "
    "otherwise hold TLT 60%+IEF 37% defensive blend; biweekly rebalance; VWO+VEA signal-only."
)

STRATEGY = EmDmRatioSp500Gate()
