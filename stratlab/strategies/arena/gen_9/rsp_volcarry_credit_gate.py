"""RSP Equal-Weight SP500 with Vol Carry + JNK Credit Gate — gen_9 sonnet-9

Hypothesis: Use RSP (equal-weight S&P500 ETF) as the primary risk vehicle,
conditioned on two regime signals:
  Signal 1: SPY realized vol carry — 21d realized vol vs 90d rolling percentile
  Signal 2: JNK credit trend — JNK vs 30d SMA

3-tier allocation:
  Vol calm (below 67th pct) AND JNK risk-on → RSP at 90% (full breadth exposure)
  Vol calm but JNK risk-off → SPY at 75% (less volatile, cap-weighted)
  Vol stressed (above 67th pct) → SPY at 55% + TLT at 42% (de-risk)
  RSP below 150d SMA (bear) → TLT at 97% (full defensive)

Rationale:
- RSP (equal-weight S&P500) has structurally different returns vs cap-weighted
  SPY: overweights small/mid-cap stocks, less tech concentration. In IS (2010-
  2018), equal-weight captured broader market breadth vs mega-cap concentration.
- RSP provides low structural corr to SP500 xsect strategies: corr to SPY ~0.97
  but daily return profile differs. RSP/SPY divergence captures breadth premium.
- Vol carry on SPY (not RSP) as the conditioning signal — established from
  gen7_realized_vol_carry_spy (IS Calmar 1.051). Using the same timing signal
  but routing to RSP instead of SPY.
- JNK credit as secondary gate: when credit is supportive, RSP's small/mid-cap
  tilt has higher credit sensitivity and benefits more.
- RSP 150d SMA as trend filter (not 200d): faster response to bear markets.

Differentiators vs leaderboard:
- RSP as primary vehicle (equal-weight SP500, not cap-weight SPY or xsect picks)
- Vol carry signal on RSP/SPY routing (not on individual stock selection)
- JNK credit + vol carry dual conditioning
- 150d SMA bear gate (not 200d) — distinct from all existing trend gates
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
RV_WINDOW = 21            # SPY realized vol window
PERCENTILE_WINDOW = 90    # rolling percentile window
RV_STRESSED_PCT = 67      # above this percentile: vol stressed
JNK_MA_WINDOW = 30        # JNK credit trend MA
RSP_TREND_WINDOW = 150    # RSP bear gate MA (faster than 200d)
# RSP exposures
RSP_FULL = 0.97           # full RSP when both signals risk-on
SPY_MID = 0.97            # SPY when vol calm but credit off (still equity)
SPY_STRESS = 0.65         # SPY when stressed (partial de-risk)
TLT_STRESS = 0.32         # TLT fill when stressed

_SPY = "SPY"
_RSP = "RSP"
_TLT = "TLT"
_JNK = "JNK"


class RspVolCarryCreditGate(Strategy):
    """RSP vol-carry + JNK credit gate: RSP 90% / SPY 75% / SPY+TLT / TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rv_window: int = RV_WINDOW,
        percentile_window: int = PERCENTILE_WINDOW,
        rv_stressed_pct: float = RV_STRESSED_PCT,
        jnk_ma_window: int = JNK_MA_WINDOW,
        rsp_trend_window: int = RSP_TREND_WINDOW,
        rsp_full: float = RSP_FULL,
        spy_mid: float = SPY_MID,
        spy_stress: float = SPY_STRESS,
        tlt_stress: float = TLT_STRESS,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rv_window=rv_window,
            percentile_window=percentile_window,
            rv_stressed_pct=rv_stressed_pct,
            jnk_ma_window=jnk_ma_window,
            rsp_trend_window=rsp_trend_window,
            rsp_full=rsp_full,
            spy_mid=spy_mid,
            spy_stress=spy_stress,
            tlt_stress=tlt_stress,
        )
        self.rebalance_every = int(rebalance_every)
        self.rv_window = int(rv_window)
        self.percentile_window = int(percentile_window)
        self.rv_stressed_pct = float(rv_stressed_pct)
        self.jnk_ma_window = int(jnk_ma_window)
        self.rsp_trend_window = int(rsp_trend_window)
        self.rsp_full = float(rsp_full)
        self.spy_mid = float(spy_mid)
        self.spy_stress = float(spy_stress)
        self.tlt_stress = float(tlt_stress)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.rsp_trend_window,
                     self.percentile_window + self.rv_window) + 10
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

        # --- RSP 150d trend gate → TLT (bear market) ---
        rsp_bull = True
        try:
            rsp_hist = ctx.history(_RSP)
            rsp_c = rsp_hist["close"].dropna()
            if len(rsp_c) >= self.rsp_trend_window + 2:
                rsp_sma = float(rsp_c.iloc[-self.rsp_trend_window:].mean())
                rsp_now = float(rsp_c.iloc[-1])
                rsp_bull = rsp_now > rsp_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not rsp_bull:
            if _TLT in live:
                target[_TLT] = 0.97
        else:
            # --- SPY realized vol carry ---
            vol_stressed = False
            try:
                spy_hist = ctx.history(_SPY)
                spy_c = spy_hist["close"].dropna()
                need = self.percentile_window + self.rv_window + 5
                if len(spy_c) >= need:
                    log_rets = np.log(spy_c.values[1:] / spy_c.values[:-1])
                    current_rv = float(np.std(log_rets[-self.rv_window:]) * np.sqrt(252))

                    rv_series = []
                    for i in range(self.percentile_window):
                        end_i = len(log_rets) - i
                        start_i = end_i - self.rv_window
                        if start_i < 0:
                            break
                        rv_series.append(float(np.std(log_rets[start_i:end_i]) * np.sqrt(252)))

                    if rv_series and np.isfinite(current_rv):
                        stressed_threshold = float(
                            np.percentile(rv_series, self.rv_stressed_pct)
                        )
                        vol_stressed = current_rv >= stressed_threshold
            except Exception:
                pass

            # --- JNK credit gate ---
            jnk_ron = True
            try:
                jnk_hist = ctx.history(_JNK)
                jnk_c = jnk_hist["close"].dropna()
                if len(jnk_c) >= self.jnk_ma_window + 2:
                    jnk_ma = float(jnk_c.iloc[-self.jnk_ma_window:].mean())
                    jnk_now = float(jnk_c.iloc[-1])
                    jnk_ron = jnk_now > jnk_ma
            except Exception:
                pass

            # --- 3-tier routing ---
            if vol_stressed:
                # Vol stressed: SPY + TLT (reduce risk regardless of credit)
                if _SPY in live:
                    target[_SPY] = self.spy_stress
                if _TLT in live:
                    target[_TLT] = self.tlt_stress
            elif not jnk_ron:
                # Vol calm but credit widening: SPY (intermediate risk)
                if _SPY in live:
                    target[_SPY] = self.spy_mid
            else:
                # Vol calm AND credit risk-on: RSP (full breadth exposure)
                if _RSP in live:
                    target[_RSP] = self.rsp_full

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


UNIVERSE = [_SPY, _RSP, _TLT, _JNK]

NAME = "rsp_volcarry_credit_gate"
HYPOTHESIS = (
    "RSP equal-weight SP500 with vol carry + JNK credit gate: RSP 90% when vol calm "
    "(SPY 21d RV below 90d 67th pct) AND JNK above 30d SMA; SPY 75% when JNK risk-off; "
    "SPY 55%+TLT 42% when vol stressed; TLT 97% when RSP below 150d SMA; weekly rebalance."
)

STRATEGY = RspVolCarryCreditGate()
