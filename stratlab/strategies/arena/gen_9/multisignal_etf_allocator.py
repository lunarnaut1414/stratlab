"""QQQ Realized Vol Carry with JNK Credit Tilt — gen_9 sonnet-9

Hypothesis: Extend the gen7 realized vol carry framework (IS Calmar 1.051,
OOS 0.447) from SPY to QQQ, with an added JNK credit tilt:
  - Primary vehicle: QQQ (not SPY) — captures tech/growth premium
  - Vol regime: QQQ 21d realized vol vs 90d rolling percentiles
    - Calm (below 33rd pct): QQQ at 90% exposure
    - Middle: QQQ at 70%
    - Stressed (above 67th pct): QQQ 50% + TLT 47%
  - JNK credit tilt overlay: when JNK above 30d SMA (credit risk-on),
    add +7% exposure to QQQ in each tier (up to 97%)
  - SPY 200d outer bear gate → TLT

Rationale:
- gen7_realized_vol_carry_spy had IS Calmar 1.051 and corr 0.7749 to top-5.
  Using QQQ instead of SPY captures the tech-growth premium in bull vol regimes.
- QQQ is ~1.2-1.5x SPY vol, so vol-targeting on QQQ is naturally more
  conservative than SPY-based carry (each tier gets modestly lower exposure).
- JNK tilt is an additional boost: when credit is healthy, lean harder into QQQ.
  When credit weakens, the base allocation is already more conservative.
- The combination of QQQ + vol carry + credit tilt is not on the leaderboard.

Differentiators vs existing strategies:
- QQQ as primary risk vehicle (not SPY or SP500 stocks)
- Realized vol carry (not VIX level threshold) — distinct from VIX-gated strats
- JNK credit tilt ON TOP of vol regime (layered signals, not OR/AND)
- Not a simple ETF allocator: vol regime primary, credit secondary
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
RV_WINDOW = 21            # QQQ realized vol window
MEDIAN_WINDOW = 90        # rolling percentile window
JNK_MA_WINDOW = 30        # JNK credit trend window
SPY_TREND_WINDOW = 200    # SPY outer bear gate
# Base QQQ exposures by vol regime
QQQ_CALM = 0.90           # below 33rd pct
QQQ_MID = 0.70            # between 33rd and 67th
QQQ_STRESS = 0.50         # above 67th
TLT_STRESS = 0.47         # TLT fill when stressed
JNK_BOOST = 0.07          # extra QQQ when JNK risk-on

_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_JNK = "JNK"


class QqqRealizedVolCarry(Strategy):
    """QQQ vol-carry with JNK credit tilt: QQQ 90/70/50% by vol regime + JNK boost."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rv_window: int = RV_WINDOW,
        median_window: int = MEDIAN_WINDOW,
        jnk_ma_window: int = JNK_MA_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        qqq_calm: float = QQQ_CALM,
        qqq_mid: float = QQQ_MID,
        qqq_stress: float = QQQ_STRESS,
        tlt_stress: float = TLT_STRESS,
        jnk_boost: float = JNK_BOOST,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rv_window=rv_window,
            median_window=median_window,
            jnk_ma_window=jnk_ma_window,
            spy_trend_window=spy_trend_window,
            qqq_calm=qqq_calm,
            qqq_mid=qqq_mid,
            qqq_stress=qqq_stress,
            tlt_stress=tlt_stress,
            jnk_boost=jnk_boost,
        )
        self.rebalance_every = int(rebalance_every)
        self.rv_window = int(rv_window)
        self.median_window = int(median_window)
        self.jnk_ma_window = int(jnk_ma_window)
        self.spy_trend_window = int(spy_trend_window)
        self.qqq_calm = float(qqq_calm)
        self.qqq_mid = float(qqq_mid)
        self.qqq_stress = float(qqq_stress)
        self.tlt_stress = float(tlt_stress)
        self.jnk_boost = float(jnk_boost)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window, self.median_window + self.rv_window) + 10
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

        # --- SPY 200d outer bear gate → TLT ---
        try:
            spy_hist = ctx.history(_SPY)
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

        target: dict[str, float] = {}

        if not spy_bull:
            if _TLT in live:
                target[_TLT] = 0.97
        else:
            # --- QQQ realized vol vs rolling percentiles ---
            qqq_exp = self.qqq_mid  # default middle
            tlt_exp = 0.0

            try:
                qqq_hist = ctx.history(_QQQ)
                qqq_c = qqq_hist["close"].dropna()
                need = self.median_window + self.rv_window + 5
                if len(qqq_c) >= need:
                    log_rets = np.log(qqq_c.values[1:] / qqq_c.values[:-1])

                    # Current 21d RV (annualized)
                    current_rv = float(np.std(log_rets[-self.rv_window:]) * np.sqrt(252))

                    # Rolling 21d RV series for last 90 bars
                    rv_series = []
                    for i in range(self.median_window):
                        end_i = len(log_rets) - i
                        start_i = end_i - self.rv_window
                        if start_i < 0:
                            break
                        rv_series.append(float(np.std(log_rets[start_i:end_i]) * np.sqrt(252)))

                    if rv_series and np.isfinite(current_rv):
                        p33 = float(np.percentile(rv_series, 33))
                        p67 = float(np.percentile(rv_series, 67))

                        if current_rv <= p33:
                            qqq_exp = self.qqq_calm
                            tlt_exp = 0.0
                        elif current_rv >= p67:
                            qqq_exp = self.qqq_stress
                            tlt_exp = self.tlt_stress
                        else:
                            qqq_exp = self.qqq_mid
                            tlt_exp = 0.0
            except Exception:
                pass

            # --- JNK credit tilt overlay ---
            jnk_ron = False
            try:
                jnk_hist = ctx.history(_JNK)
                jnk_c = jnk_hist["close"].dropna()
                if len(jnk_c) >= self.jnk_ma_window + 2:
                    jnk_ma = float(jnk_c.iloc[-self.jnk_ma_window:].mean())
                    jnk_now = float(jnk_c.iloc[-1])
                    jnk_ron = jnk_now > jnk_ma
            except Exception:
                pass

            # Apply JNK boost: when credit healthy, add to QQQ, cap at 0.97
            if jnk_ron:
                qqq_exp = min(qqq_exp + self.jnk_boost, 0.97)

            if _QQQ in live:
                target[_QQQ] = qqq_exp
            if tlt_exp > 0 and _TLT in live:
                target[_TLT] = tlt_exp

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


UNIVERSE = [_SPY, _QQQ, _TLT, _JNK]

NAME = "multisignal_etf_allocator"
HYPOTHESIS = (
    "QQQ realized vol carry with JNK credit tilt: QQQ exposure scaled by 21d realized vol "
    "vs 90d rolling percentiles (calm=90%, mid=70%, stressed=50%+TLT 47%); +7% QQQ boost "
    "when JNK above 30d SMA; SPY 200d outer bear gate to TLT; weekly rebalance."
)

STRATEGY = QqqRealizedVolCarry()
