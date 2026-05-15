"""gen_9 sonnet-1 — GDX/IAU Miners Signal × SPY Vol-Carry Sizing

Hypothesis: Extend the realized-vol carry concept (gen7_realized_vol_carry_spy,
IS 1.05, OOS 0.45) by combining TWO signals for SPY exposure sizing:
1. SPY realized vol 21d vs 63d distribution (existing vol-carry signal)
2. GDX vs IAU 21d return (commodity cycle signal — miners leading gold = expansion)

Both signals must agree for maximum exposure:
- Vol calm AND miners lead gold → SPY 92% (max)
- Vol calm OR miners lead gold (only one) → SPY 78%
- Vol stressed AND miners lag gold → SPY 55%
- SPY 200d SMA outer bear gate → TLT 97%

Rationale:
- gen7_realized_vol_carry_spy showed vol-percentile-based SPY sizing produces
  strong IS Calmar (1.05) with stable halves and OOS survival (0.45).
- Adding the GDX/IAU commodity cycle signal provides a second orthogonal
  dimension: equity risk appetite driven by real-asset demand vs safe-haven gold.
- Combined signal should improve timing — avoid reducing exposure just because
  vol is high if commodity cycle confirms expansion; also reduce exposure if
  miners underperform even during calm vol periods (advance warning).
- All-SPY portfolio means very low corr to SP500 individual stock selectors
  (different risk profile) while avoiding TLT correlation issues (bonds only
  in bear regime).

Coverage (all cover IS 2010-2018):
  SPY (1993), TLT (2002), GDX (2006), IAU (2005), ^VIX not needed (use realized vol)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

RV_WINDOW = 21          # realized vol window (days)
DIST_WINDOW = 63        # distribution window for percentile
MINERS_WINDOW = 21      # GDX vs IAU 21d return
SPY_TREND = 200         # 200d SMA outer bear gate
REBALANCE_EVERY = 5     # weekly

EXP_MAX = 0.92          # both calm + miners-on
EXP_MID = 0.78          # one signal on
EXP_MIN = 0.55          # both stressed

_SPY = "SPY"
_TLT = "TLT"
_GDX = "GDX"
_IAU = "IAU"


class GdxSpyVolCarry(Strategy):
    """GDX/IAU miners signal + SPY realized-vol carry for SPY exposure tiers."""

    def __init__(
        self,
        rv_window: int = RV_WINDOW,
        dist_window: int = DIST_WINDOW,
        miners_window: int = MINERS_WINDOW,
        spy_trend: int = SPY_TREND,
        rebalance_every: int = REBALANCE_EVERY,
        exp_max: float = EXP_MAX,
        exp_mid: float = EXP_MID,
        exp_min: float = EXP_MIN,
    ) -> None:
        super().__init__(
            rv_window=rv_window,
            dist_window=dist_window,
            miners_window=miners_window,
            spy_trend=spy_trend,
            rebalance_every=rebalance_every,
            exp_max=exp_max,
            exp_mid=exp_mid,
            exp_min=exp_min,
        )
        self.rv_window = int(rv_window)
        self.dist_window = int(dist_window)
        self.miners_window = int(miners_window)
        self.spy_trend = int(spy_trend)
        self.rebalance_every = int(rebalance_every)
        self.exp_max = float(exp_max)
        self.exp_mid = float(exp_mid)
        self.exp_min = float(exp_min)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend, self.dist_window + self.rv_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
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
            # Bear market → TLT
            if _TLT in live:
                target[_TLT] = self.exp_max
        else:
            # --- SPY realized vol percentile (vol-carry signal) ---
            vol_calm = True  # default: calm
            try:
                need = self.dist_window + self.rv_window + 5
                prices_window = ctx.closes_window(need)
                if _SPY in prices_window.columns:
                    spy_prices = prices_window[_SPY].dropna()
                    if len(spy_prices) >= self.dist_window + self.rv_window:
                        log_rets = np.log(spy_prices.values[1:] / spy_prices.values[:-1])
                        # Current RV (21d)
                        rv_current = float(np.std(log_rets[-self.rv_window:])) * np.sqrt(252)
                        # Historical RV distribution (63d window of 21d RVs)
                        rv_hist = []
                        for i in range(self.dist_window):
                            start_i = -(self.rv_window + i + 1)
                            end_i = -(i + 1) if i > 0 else None
                            window = log_rets[start_i:end_i]
                            if len(window) >= self.rv_window // 2:
                                rv_hist.append(float(np.std(window)) * np.sqrt(252))
                        if rv_hist:
                            pctile = float(np.mean(np.array(rv_hist) >= rv_current))
                            vol_calm = pctile >= 0.50  # calm = below-median RV
            except Exception:
                pass

            # --- GDX vs IAU miners signal ---
            miners_on = True  # default: miners-on
            try:
                m_need = self.miners_window + 5
                m_prices = ctx.closes_window(m_need)
                if _GDX in m_prices.columns and _IAU in m_prices.columns:
                    gdx_col = m_prices[_GDX].dropna()
                    iau_col = m_prices[_IAU].dropna()
                    if len(gdx_col) >= self.miners_window and len(iau_col) >= self.miners_window:
                        gdx_ret = float(gdx_col.iloc[-1] / gdx_col.iloc[-self.miners_window] - 1.0)
                        iau_ret = float(iau_col.iloc[-1] / iau_col.iloc[-self.miners_window] - 1.0)
                        if np.isfinite(gdx_ret) and np.isfinite(iau_ret):
                            miners_on = gdx_ret > iau_ret
            except Exception:
                pass

            # Determine SPY exposure
            if vol_calm and miners_on:
                exposure = self.exp_max
            elif vol_calm or miners_on:
                exposure = self.exp_mid
            else:
                exposure = self.exp_min

            if _SPY in live:
                target[_SPY] = exposure

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


UNIVERSE = [_SPY, _TLT, _GDX, _IAU]

NAME = "gdx_spy_vol_carry"
HYPOTHESIS = (
    "SPY vol-carry sizing extended with GDX/IAU miners signal: "
    "both vol calm (21d RV below 63d median) AND miners lead gold (GDX>IAU 21d) → SPY 92%; "
    "one signal on → SPY 78%; both stressed → SPY 55%; "
    "SPY 200d bear → TLT; weekly rebalance; "
    "vol-carry + commodity cycle dual-signal SPY tilt distinct from VIX-level or JNK gates"
)

STRATEGY = GdxSpyVolCarry()
