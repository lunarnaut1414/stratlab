"""gen_9 sonnet-1 — Sector/Industry ETF Rotation with VIX Percentile Gate

Hypothesis: Rank 8 sector and industry ETFs (XLE, XLF, XLK, XLU, XLI, XLY,
XBI, KRE) by 42d momentum; hold top-3 when VIX 63d rolling-percentile rank
is below 50th percentile (calm market); rotate all to TLT when VIX percentile
is above 70th percentile (stressed market); intermediate regime holds top-2
ETFs with reduced exposure. SPY 200d SMA outer bear gate. Biweekly rebalance.

Differentiation rationale:
- Uses VIX PERCENTILE rank (not absolute threshold) — avoids OOS fragility of
  absolute thresholds seen in gen_7/8 wildcard failures.
- Entirely ETF-based (no individual stock selection) — structurally distinct from
  SP500 cross-sectional cluster that dominates the top-5 leaderboard.
- Sub-sector mix (XBI biotech, KRE regional banks) goes beyond SPDR XL* rotation
  that prior rounds covered.
- VIX percentile gate adapts dynamically to changing vol regimes vs fixed thresholds.

Coverage check (all cover IS 2010-2018):
  XLE (1998), XLF (1998), XLK (1998), XLU (1998), XLI (1998), XLY (1998),
  XBI (2006), KRE (2006), TLT (2002), ^VIX (1990)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# Tradeable sector/industry ETFs (all covering IS window 2010-2018)
SECTOR_ETFS = ["XLE", "XLF", "XLK", "XLU", "XLI", "XLY", "XBI", "KRE"]
DEFENSIVE_ETF = "TLT"
MARKET_ETF = "SPY"
VIX_SYM = "^VIX"

MOMENTUM_WINDOW = 42         # 42-day sector momentum
VIX_PCTILE_WINDOW = 63       # rolling 63d VIX distribution
SPY_TREND_WINDOW = 200       # 200d SMA bear gate
TOP_K_CALM = 3               # hold top-3 when VIX calm
TOP_K_NEUTRAL = 2            # hold top-2 in neutral regime
CALM_PCTILE = 50             # VIX below 50th pctile = calm
STRESS_PCTILE = 70           # VIX above 70th pctile = stressed
EXPOSURE = 0.97
REBALANCE_EVERY = 10         # biweekly


class SectorVixPctileRotation(Strategy):
    """Sector/industry ETF momentum with VIX percentile-rank regime gate."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        vix_pctile_window: int = VIX_PCTILE_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k_calm: int = TOP_K_CALM,
        top_k_neutral: int = TOP_K_NEUTRAL,
        calm_pctile: float = CALM_PCTILE,
        stress_pctile: float = STRESS_PCTILE,
        exposure: float = EXPOSURE,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            vix_pctile_window=vix_pctile_window,
            spy_trend_window=spy_trend_window,
            top_k_calm=top_k_calm,
            top_k_neutral=top_k_neutral,
            calm_pctile=calm_pctile,
            stress_pctile=stress_pctile,
            exposure=exposure,
            rebalance_every=rebalance_every,
        )
        self.momentum_window = int(momentum_window)
        self.vix_pctile_window = int(vix_pctile_window)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k_calm = int(top_k_calm)
        self.top_k_neutral = int(top_k_neutral)
        self.calm_pctile = float(calm_pctile)
        self.stress_pctile = float(stress_pctile)
        self.exposure = float(exposure)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.vix_pctile_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(MARKET_ETF)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- VIX percentile rank ---
        vix_pctile = 50.0  # default: neutral
        try:
            vix_hist = ctx.history(VIX_SYM)
            if vix_hist is not None and len(vix_hist) >= self.vix_pctile_window + 2:
                vix_close = vix_hist["close"].dropna()
                if len(vix_close) >= self.vix_pctile_window + 1:
                    window = vix_close.values[-self.vix_pctile_window:]
                    vix_now = float(vix_close.iloc[-1])
                    vix_pctile = float(np.mean(window <= vix_now) * 100.0)
        except Exception:
            pass

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull or vix_pctile >= self.stress_pctile:
            # Bear market or VIX stressed → fully defensive TLT
            if DEFENSIVE_ETF in live:
                target[DEFENSIVE_ETF] = self.exposure
        else:
            # Rank sector ETFs by momentum
            scores: dict[str, float] = {}
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)

            for sym in SECTOR_ETFS:
                if sym not in prices.columns:
                    continue
                col = prices[sym].dropna()
                if len(col) < self.momentum_window:
                    continue
                ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                if np.isfinite(ret):
                    scores[sym] = ret

            if not scores:
                if DEFENSIVE_ETF in live:
                    target[DEFENSIVE_ETF] = self.exposure
            else:
                ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                # Determine top_k based on VIX regime
                if vix_pctile < self.calm_pctile:
                    top_k = self.top_k_calm       # calm: top-3
                    exp = self.exposure
                else:
                    top_k = self.top_k_neutral    # neutral: top-2, reduced
                    exp = self.exposure * 0.80

                selected = [s for s in ranked[:top_k] if s in live]
                if not selected:
                    if DEFENSIVE_ETF in live:
                        target[DEFENSIVE_ETF] = self.exposure
                else:
                    per_weight = exp / len(selected)
                    for sym in selected:
                        target[sym] = per_weight

        orders: list[Order] = []
        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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


UNIVERSE = SECTOR_ETFS + [DEFENSIVE_ETF, MARKET_ETF, VIX_SYM]

NAME = "sector_vix_pctile_rotation"
HYPOTHESIS = (
    "Sub-sector ETF momentum rotation: rank XBI, XLE, XLF, XLK, XLU, XLI, XLY, KRE by "
    "42d return; hold top-3 when VIX 63d rolling percentile rank <50th pct (calm); "
    "top-2 at 80% exposure in neutral regime; TLT when VIX >70th pct or SPY bear; "
    "biweekly rebalance; percentile-rank VIX gate avoids absolute-threshold OOS fragility"
)

STRATEGY = SectorVixPctileRotation()
