"""USD-Strength Macro Sector Rotation — gen_8 sonnet-9

Hypothesis: Use UUP (PowerShares USD Index ETF) 20-day vs 60-day SMA
crossover as the USD-regime signal.

- Rising USD (UUP 20d > 60d MA): defensives regime; hold XLU 34% + XLP 33%
  + TLT 30% (utilities, staples, long bonds all benefit from strong dollar).
- Falling USD (UUP 20d < 60d MA): cyclicals regime; hold XLB 34% + XLE 33%
  + XLK 30% if each sector ETF is above its 50d SMA, else substitute with IEF.
- Neutral / transition (within 0.3% of each other): hold SPY 97%.
- SPY 200d SMA bear override: hold IEF 60% + TLT 37%.

Weekly rebalance. The dollar-strength signal creates a novel macro axis
orthogonal to VIX (risk appetite), JNK (credit), and TNX (rate direction).

Rationale: USD strength historically pressures commodity exporters (XLE, XLB)
and foreign earnings (tech multinationals), while supporting utilities
(domestic revenue, lower input costs) and long bonds (dollar-positive carry).
This creates a cross-asset rotation that is distinct from all existing
leaderboard signals.

Differentiation: Sonnet-9 of gen_5 tried UUP-based EEM rotation
(gen6_dollar_strength_em_rotation) which was abandoned at IS Calmar 0.400.
That approach routed to EEM (EM equities) in falling-dollar, which struggled
because EEM = foreign market index. THIS strategy routes entirely to US sector
ETFs in both regimes, avoiding the EM data quality and correlation issues.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UUP_FAST = 20             # Fast MA for USD signal
UUP_SLOW = 60             # Slow MA for USD signal
TREND_WINDOW = 200        # SPY 200d SMA bear gate
SECTOR_TREND = 50         # Sector ETF 50d SMA check
REBALANCE_DAYS = 5        # Weekly
EXPOSURE = 0.97

UNIVERSE = [
    "UUP",           # USD index ETF (signal)
    "SPY",           # Benchmark + bear gate
    "IEF",           # Intermediate bonds (defensive fallback)
    "TLT",           # Long bonds (defensive)
    "XLU",           # Utilities (rising USD, defensive)
    "XLP",           # Consumer staples (rising USD, defensive)
    "XLB",           # Materials (falling USD, cyclical)
    "XLE",           # Energy (falling USD, cyclical)
    "XLK",           # Tech (falling USD, growth)
]


class UsdMacroSectorRotation(Strategy):
    """USD-strength macro sector rotation using UUP SMA crossover."""

    def __init__(self, **params: float) -> None:
        super().__init__(**params)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(TREND_WINDOW, UUP_SLOW) + 5
        if ctx.idx < warmup:
            return []

        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []

        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # --- SPY 200d SMA bear gate ---
        spy_hist = ctx.history("SPY")
        spy_bear = False
        if len(spy_hist) >= TREND_WINDOW:
            spy_sma = float(spy_hist["close"].iloc[-TREND_WINDOW:].mean())
            spy_price = live.get("SPY", 0.0)
            spy_bear = (spy_price > 0) and (spy_price <= spy_sma)

        if spy_bear:
            # Bear market: IEF + TLT defensive blend
            target = {"IEF": 0.60, "TLT": 0.37}
        else:
            # --- UUP 20d vs 60d SMA crossover ---
            uup_hist = ctx.history("UUP")
            if len(uup_hist) < UUP_SLOW:
                return []

            uup_close = uup_hist["close"]
            uup_fast_sma = float(uup_close.iloc[-UUP_FAST:].mean())
            uup_slow_sma = float(uup_close.iloc[-UUP_SLOW:].mean())

            # Threshold: 0.3% band to avoid whipsawing in neutral
            neutral_band = 0.003 * uup_slow_sma
            uup_spread = uup_fast_sma - uup_slow_sma

            if abs(uup_spread) <= neutral_band:
                # Neutral: hold SPY
                target = {"SPY": EXPOSURE}
            elif uup_spread > neutral_band:
                # Rising USD: defensives
                target = {"XLU": 0.34, "XLP": 0.33, "TLT": 0.30}
            else:
                # Falling USD: cyclicals with sector trend gate
                cyclicals = []
                for sym in ["XLB", "XLE", "XLK"]:
                    hist = ctx.history(sym)
                    if len(hist) < SECTOR_TREND:
                        cyclicals.append(("IEF", sym))
                        continue
                    sma = float(hist["close"].iloc[-SECTOR_TREND:].mean())
                    price = live.get(sym, 0.0)
                    if price > 0 and price > sma:
                        cyclicals.append((sym, None))
                    else:
                        cyclicals.append(("IEF", sym))  # Substitute with IEF

                # Build weights: 34/33/30 across 3 slots
                weights = [0.34, 0.33, 0.30]
                target: dict[str, float] = {}
                for (sym, _), w in zip(cyclicals, weights):
                    if sym in target:
                        target[sym] += w
                    else:
                        target[sym] = w

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                orders.append(Order(side=OrderSide.SELL, size=abs(pos.size), symbol=sym))

        # Build target positions
        for sym, weight in target.items():
            price = live.get(sym, 0.0)
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


NAME = "usd_macro_sector_rotation"
HYPOTHESIS = (
    "USD-strength macro rotation: UUP 20d vs 60d SMA crossover; rising USD hold XLU+XLP+TLT; "
    "falling USD hold XLB+XLE+XLK (50d SMA gate, else IEF substitute); neutral hold SPY 97%; "
    "SPY 200d bear gate to IEF+TLT; weekly rebalance; dollar-strength macro signal novel vs leaderboard."
)

STRATEGY = UsdMacroSectorRotation()
