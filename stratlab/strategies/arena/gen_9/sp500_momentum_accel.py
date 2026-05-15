"""gen_9 sonnet-3 — SP500 Momentum Acceleration

Hypothesis: Rank SP500 stocks by MOMENTUM ACCELERATION: the difference between
21d return and 63d return (21d outperformance vs medium-term baseline).

Positive acceleration = stock is picking up speed relative to its recent trend.
This captures stocks at INFLECTION POINTS — transitioning from flat/rising to
strongly rising — rather than established winners.

Filters applied on top of acceleration ranking:
  - Price above 200d SMA (trend confirmation)
  - Positive 63d absolute return (not just relatively improving; genuine bull)
  - JNK 30d SMA credit gate (credit risk-on required for stock selection)
  - SPY 200d SMA bear gate to TLT

Inverse-vol weighting reduces concentration in high-volatility accelerators.
Biweekly rebalance (10 bars).

Rationale: Standard momentum strategies capture LEVELS of momentum; acceleration
captures the CHANGE in momentum — a second-order signal. Stocks with high raw
momentum are often already crowded; stocks with high acceleration may represent
under-discovered opportunities with improving momentum.

Distinction from existing leaderboard:
  - gen7_sp500_idiosyncratic_momentum: ranks by beta-adjusted alpha (first-order)
  - gen8_sp500_skipmon_63sma_momentum: skip-month returns (first-order)
  - gen6_nearhi_momentum_quality: 126d return + nearhi filter (first-order)
  - No strategy ranks by 21d-minus-63d ACCELERATION spread
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
SHORT_MOM = 21          # short-term momentum
MED_MOM = 63            # medium-term baseline
TREND_WINDOW = 200      # SPY 200d SMA
STOCK_TREND_WINDOW = 200  # per-stock 200d SMA
JNK_MA_WINDOW = 30
TOP_K = 15
VOL_WINDOW = 21
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_JNK = "JNK"


class Sp500MomentumAccel(Strategy):
    """SP500 momentum acceleration: 21d return minus 63d return as ranking signal."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        short_mom: int = SHORT_MOM,
        med_mom: int = MED_MOM,
        trend_window: int = TREND_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        jnk_ma_window: int = JNK_MA_WINDOW,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            short_mom=short_mom,
            med_mom=med_mom,
            trend_window=trend_window,
            stock_trend_window=stock_trend_window,
            jnk_ma_window=jnk_ma_window,
            top_k=top_k,
            vol_window=vol_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.short_mom = int(short_mom)
        self.med_mom = int(med_mom)
        self.trend_window = int(trend_window)
        self.stock_trend_window = int(stock_trend_window)
        self.jnk_ma_window = int(jnk_ma_window)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_trend_window,
                     self.med_mom, self.jnk_ma_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # --- SPY 200d SMA bear gate ---
        spy_bull = True
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_now = float(spy_close.iloc[-1])
                    spy_bull = spy_now > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # SPY bear — TLT defensive
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # --- JNK credit gate ---
            credit_risk_on = True
            try:
                jnk_hist = ctx.history(_JNK)
                if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma_window + 2:
                    jnk_close = jnk_hist["close"].dropna()
                    if len(jnk_close) >= self.jnk_ma_window:
                        jnk_sma = float(jnk_close.iloc[-self.jnk_ma_window:].mean())
                        jnk_now = float(jnk_close.iloc[-1])
                        credit_risk_on = jnk_now > jnk_sma
            except Exception:
                pass

            if not credit_risk_on:
                # Credit weak — TLT defensive
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                # Credit + SPY bull — compute momentum acceleration
                need = max(self.med_mom, self.stock_trend_window) + 10
                prices = ctx.closes_window(need)
                if len(prices) < self.med_mom:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    accel_scores: dict[str, float] = {}
                    med_rets: dict[str, float] = {}
                    smas: dict[str, float] = {}

                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _JNK):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.med_mom + 5:
                            continue

                        p_now = float(col.iloc[-1])
                        if p_now <= 0:
                            continue

                        # Short-term 21d return
                        if len(col) >= self.short_mom:
                            p_21 = float(col.iloc[-self.short_mom])
                            if p_21 > 0:
                                r_short = p_now / p_21 - 1.0
                            else:
                                continue
                        else:
                            continue

                        # Medium-term 63d return
                        if len(col) >= self.med_mom:
                            p_63 = float(col.iloc[-self.med_mom])
                            if p_63 > 0:
                                r_med = p_now / p_63 - 1.0
                            else:
                                continue
                        else:
                            continue

                        # Acceleration = 21d return minus 63d return
                        accel = r_short - r_med
                        if not np.isfinite(accel):
                            continue

                        # Per-stock 200d SMA filter
                        if len(col) >= self.stock_trend_window:
                            sma_val = float(col.iloc[-self.stock_trend_window:].mean())
                            if p_now <= sma_val:
                                continue  # below own 200d SMA — skip
                        else:
                            continue

                        # Positive 63d absolute return (not just relatively improving)
                        if r_med <= 0:
                            continue

                        if sym in live:
                            accel_scores[sym] = accel
                            med_rets[sym] = r_med

                    if len(accel_scores) < 5:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        ranked = sorted(accel_scores, key=accel_scores.__getitem__, reverse=True)
                        # Take top-K candidates
                        candidates = ranked[:self.top_k]

                        # Inverse-vol weighting
                        inv_vols: dict[str, float] = {}
                        for sym in candidates:
                            try:
                                s_hist = ctx.history(sym)
                                if s_hist is None:
                                    continue
                                s_close = s_hist["close"].dropna()
                                if len(s_close) >= self.vol_window + 1:
                                    rets = s_close.iloc[-(self.vol_window + 1):].pct_change().dropna()
                                    rv = float(rets.std()) * np.sqrt(252)
                                else:
                                    rv = 0.20
                            except Exception:
                                rv = 0.20
                            if rv <= 0:
                                rv = 0.20
                            if sym in live:
                                inv_vols[sym] = 1.0 / rv

                        if not inv_vols:
                            if _SPY in live:
                                target[_SPY] = self.exposure
                        else:
                            total = sum(inv_vols.values())
                            for sym, iv in inv_vols.items():
                                target[sym] = self.exposure * iv / total

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + [_TLT, _SPY, _JNK]


UNIVERSE = _universe

NAME = "sp500_momentum_accel"
HYPOTHESIS = (
    "SP500 momentum acceleration: rank stocks by 21d return minus 63d return "
    "(acceleration of momentum, positive = recent outperformance vs medium-term); "
    "hold top-15 above 200d SMA with positive 63d return (absolute momentum filter); "
    "JNK 30d SMA credit gate; inverse-vol weighted; SPY 200d bear gate to TLT; "
    "biweekly rebalance; acceleration of momentum as ranking signal not in leaderboard"
)

STRATEGY = Sp500MomentumAccel()
