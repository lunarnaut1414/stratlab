"""EM vs DM Relative Strength Z-Score SP500 Gate — gen_9 sonnet-2

Hypothesis: The VWO/VEA price ratio captures relative performance of emerging
markets vs developed markets. When EM leads DM (high z-score), it signals
global risk appetite — investors are willing to take on currency and EM
political risk, typically in environments with synchronized global growth and
dollar weakness. When DM leads EM (low z-score), it signals flight-to-quality
within international equities, often coinciding with domestic US equity stress.

Three-tier regime using 90-day z-score of the VWO/VEA ratio:
  - High z-score (>+0.5): EM outperforming DM → global risk-on →
    hold top-15 SP500 stocks by 63d momentum above 200d SMA at 97%.
  - Low z-score (<-0.5): DM outperforming EM → flight-to-quality →
    hold TLT 97% (full defensive).
  - Neutral (-0.5 to +0.5): hold SPY 60% + IEF 37%.

SPY 200d SMA outer bear gate: if SPY below 200d SMA, force TLT regardless.

Differentiation from leaderboard:
  - gen7_opus2_vea_vwo_signal_us_only uses VWO/VEA as a 60d ratio comparison
    (simple 60-day return difference), not a z-score. It also uses a simple
    binary: VWO > VEA => SP500 stocks, not a 3-tier regime.
  - This strategy uses a rolling 90-day z-score of the VWO/VEA PRICE ratio
    (not a return comparison), providing a self-adjusting normalization that
    distinguishes a small outperformance in a high-vol period vs low-vol period.
  - The z-score approach means the signal adapts to recent EM/DM regime —
    a key distinction from the prior 60d return comparison.

VWO and VEA are signal-only inputs (not traded).
IS window: 2010-2018. VWO inception 2005, VEA inception 2007 — both cover IS.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy
from stratlab.data.universe import sp500_tickers

MOMENTUM_WINDOW = 63      # 3-month momentum for stock ranking
TREND_WINDOW = 200        # SPY 200d SMA bear gate
ZSCORE_WINDOW = 90        # Rolling window for VWO/VEA ratio z-score
Z_HIGH = 0.5              # Above: risk-on (SP500 momentum)
Z_LOW = -0.5              # Below: risk-off (TLT)
TOP_K = 15
REBALANCE_DAYS = 10       # Biweekly
EXPOSURE = 0.97


def _universe() -> list[str]:
    # VWO and VEA are used as signals only (not traded)
    return sp500_tickers() + ["VWO", "VEA", "TLT", "IEF", "SPY"]


UNIVERSE = _universe


class EmDmZscoreSp500Gate(Strategy):
    """SP500 63d momentum with VWO/VEA z-score 3-tier global risk-regime gate."""

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
            # --- Compute VWO/VEA ratio z-score ---
            vwo_hist = ctx.history("VWO")
            vea_hist = ctx.history("VEA")

            if len(vwo_hist) < ZSCORE_WINDOW + 5 or len(vea_hist) < ZSCORE_WINDOW + 5:
                return []

            vwo_close = vwo_hist["close"].tail(ZSCORE_WINDOW + 5)
            vea_close = vea_hist["close"].tail(ZSCORE_WINDOW + 5)

            min_len = min(len(vwo_close), len(vea_close))
            if min_len < ZSCORE_WINDOW:
                return []

            vwo_vals = vwo_close.values[-min_len:]
            vea_vals = vea_close.values[-min_len:]

            # VWO/VEA ratio (EM per unit of DM)
            vea_safe = np.where(vea_vals > 0, vea_vals, np.nan)
            ratio = vwo_vals / vea_safe

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
                # DM outperforming EM: flight-to-quality → TLT
                target = {"TLT": EXPOSURE}
            elif z_score <= Z_HIGH:
                # Neutral global risk: SPY + IEF blend
                target = {"SPY": 0.60, "IEF": 0.37}
            else:
                # EM outperforming DM: global risk-on → SP500 momentum
                prices_window = ctx.closes_window(MOMENTUM_WINDOW + 5)
                if len(prices_window) < MOMENTUM_WINDOW:
                    target = {"SPY": 0.60, "IEF": 0.37}
                else:
                    # Exclude signal-only ETFs from stock selection
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
                        target = {"SPY": 0.60, "IEF": 0.37}
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


NAME = "em_dm_zscore_sp500_gate"
HYPOTHESIS = (
    "EM vs DM relative strength: VWO/VEA price ratio 90d z-score as global risk-appetite gate; "
    "z>+0.5 (EM leads DM, global risk-on) hold top-15 SP500 stocks 63d momentum 97%; "
    "z<-0.5 (DM leads EM, flight-to-quality) hold TLT 97%; "
    "neutral: hold SPY 60%+IEF 37%; SPY 200d outer bear gate; biweekly rebalance."
)

STRATEGY = EmDmZscoreSp500Gate()
