"""gen_9 sonnet-3 — SP500 63d Momentum with Acceleration Confirmation Filter

Hypothesis: Rank SP500 stocks by standard 63d momentum (proven OOS signal), but apply
a momentum ACCELERATION FILTER: only select stocks where their 21d return exceeds
their 63d return (momentum is accelerating, not decelerating).

This is acceleration as a FILTER (not the ranking criterion). The ranking remains
63d momentum which is proven. The filter removes stocks whose momentum is fading
(recent 21d lags the 63d average pace).

Logic:
  - Primary rank: 63d return (standard proven momentum)
  - Acceleration filter: 21d return > 63d return * (21/63) ≈ 21d pace exceeds medium pace
    More precisely: 21d_return > 63d_return / 3 (monthly outperformance vs trimonthly average)
  - Trend filter: stock above 200d SMA
  - Credit gate: JNK 30d SMA
  - Market gate: SPY 200d SMA
  - Defensive: TLT
  - Weighting: inverse-vol

Rationale: Pure 63d momentum includes stocks that had strong momentum 2 months ago but
are now decelerating. The acceleration filter removes these "past winner" traps. Stocks
with BOTH high 63d momentum AND accelerating recent 21d pace represent momentum with
continuation energy. This reduces the strategy's exposure to late-cycle momentum traps.

Difference from pure accel ranking: corr_check showed pure accel has IS Calmar 0.35;
the issue was many high-acceleration stocks are small-cap/low-quality reversal traps.
Using 63d rank first + accel filter keeps the quality of 63d selection while adding
continuation confirmation. Different from nearhi filter (price vs high) and from
idiosyncratic (market-adjusted) — this is a pure time-series momentum persistence check.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MED_MOM = 63            # primary ranking window
SHORT_MOM = 21          # acceleration confirmation window
TREND_WINDOW = 200      # SPY 200d SMA
STOCK_TREND = 200       # per-stock trend filter
JNK_MA = 30
TOP_K = 15
VOL_WINDOW = 21
EXPOSURE = 0.97
# Acceleration threshold: 21d return must exceed (63d return / 3)
# This means 21d pace >= 63d monthly average pace
ACCEL_RATIO = 3.0  # 63d / 21d ≈ 3
_SPY = "SPY"
_TLT = "TLT"
_JNK = "JNK"


class Sp500MomAccelFilter(Strategy):
    """SP500 63d momentum ranked, filtered to accelerating-momentum stocks only."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        med_mom: int = MED_MOM,
        short_mom: int = SHORT_MOM,
        trend_window: int = TREND_WINDOW,
        stock_trend: int = STOCK_TREND,
        jnk_ma: int = JNK_MA,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        exposure: float = EXPOSURE,
        accel_ratio: float = ACCEL_RATIO,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            med_mom=med_mom,
            short_mom=short_mom,
            trend_window=trend_window,
            stock_trend=stock_trend,
            jnk_ma=jnk_ma,
            top_k=top_k,
            vol_window=vol_window,
            exposure=exposure,
            accel_ratio=accel_ratio,
        )
        self.rebalance_every = int(rebalance_every)
        self.med_mom = int(med_mom)
        self.short_mom = int(short_mom)
        self.trend_window = int(trend_window)
        self.stock_trend = int(stock_trend)
        self.jnk_ma = int(jnk_ma)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.exposure = float(exposure)
        self.accel_ratio = float(accel_ratio)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.stock_trend, self.med_mom, self.jnk_ma) + 10
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

        # --- SPY 200d bear gate ---
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
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # --- JNK credit gate ---
            credit_ok = True
            try:
                jnk_hist = ctx.history(_JNK)
                if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma + 2:
                    jnk_close = jnk_hist["close"].dropna()
                    if len(jnk_close) >= self.jnk_ma:
                        jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                        credit_ok = float(jnk_close.iloc[-1]) > jnk_sma
            except Exception:
                pass

            if not credit_ok:
                if _TLT in live:
                    target[_TLT] = self.exposure
            else:
                need = max(self.med_mom, self.stock_trend) + 10
                prices = ctx.closes_window(need)
                if len(prices) < self.med_mom:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Rank by 63d momentum, filter by acceleration
                    mom_scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _JNK):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.med_mom + 5:
                            continue

                        p_now = float(col.iloc[-1])
                        if p_now <= 0:
                            continue

                        # Per-stock 200d SMA filter
                        if len(col) >= self.stock_trend:
                            sma_val = float(col.iloc[-self.stock_trend:].mean())
                            if p_now <= sma_val:
                                continue
                        else:
                            continue

                        # 63d momentum
                        p_63 = float(col.iloc[-self.med_mom])
                        if p_63 <= 0:
                            continue
                        r_med = p_now / p_63 - 1.0
                        if not np.isfinite(r_med) or r_med <= 0:
                            continue

                        # 21d momentum (acceleration)
                        if len(col) >= self.short_mom:
                            p_21 = float(col.iloc[-self.short_mom])
                            if p_21 <= 0:
                                continue
                            r_short = p_now / p_21 - 1.0
                        else:
                            continue

                        # Acceleration filter: 21d return must be >= 63d return / ACCEL_RATIO
                        # i.e., annualized pace of 21d >= annualized pace of 63d
                        # Simple: 21d_ret >= 63d_ret / 3  (3 = 63/21 periods)
                        accel_threshold = r_med / self.accel_ratio
                        if r_short < accel_threshold:
                            continue  # momentum decelerating — skip

                        if sym in live:
                            mom_scores[sym] = r_med  # rank by 63d return (primary)

                    if len(mom_scores) < 5:
                        # Few qualifying stocks (accel filter too tight) — fall back to
                        # standard 63d momentum without accel filter
                        for sym in prices.columns:
                            if sym in (_SPY, _TLT, _JNK):
                                continue
                            col = prices[sym].dropna()
                            if len(col) < self.med_mom + 5:
                                continue
                            p_now = float(col.iloc[-1])
                            if p_now <= 0:
                                continue
                            if len(col) >= self.stock_trend:
                                sma_val = float(col.iloc[-self.stock_trend:].mean())
                                if p_now <= sma_val:
                                    continue
                            else:
                                continue
                            p_63 = float(col.iloc[-self.med_mom])
                            if p_63 <= 0:
                                continue
                            r_med = p_now / p_63 - 1.0
                            if not np.isfinite(r_med) or r_med <= 0:
                                continue
                            if sym in live and sym not in mom_scores:
                                mom_scores[sym] = r_med

                    if len(mom_scores) < 3:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        ranked = sorted(mom_scores, key=mom_scores.__getitem__, reverse=True)
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

NAME = "sp500_mom_accel_filter"
HYPOTHESIS = (
    "SP500 63d momentum with acceleration confirmation filter: rank by 63d return, "
    "but only select stocks where 21d return exceeds 63d return / 3 (momentum accelerating); "
    "hold top-15 above 200d SMA; JNK 30d SMA credit gate; inverse-vol weighted; "
    "SPY 200d bear gate to TLT; biweekly rebalance; acceleration as FILTER on top of "
    "standard ranking reduces crowding vs pure momentum"
)

STRATEGY = Sp500MomAccelFilter()
