"""QQQ with RSP Breadth Gate + VIX Realized Vol Hedge — gen_9 sonnet-9

Hypothesis: Hold QQQ 97% when RSP (equal-weight S&P500) is above its 100d SMA
(broad market participation = healthy bull) AND SPY 200d SMA is bullish.
When RSP fails 100d SMA but SPY still bullish, downgrade to SPY 97% (narrow
leadership = less tech-growth premium).
When SPY below 200d SMA, go to TLT.
Vol hedge overlay: when SPY 21d realized vol is above 90d 80th percentile,
trim QQQ/SPY to 65% and add TLT 32% (volatile periods reduce exposure).
Weekly rebalance.

Rationale:
- RSP/SPY breadth as QQQ qualification gate: QQQ outperforms most when the
  broader market is participating (RSP leading). When only mega-caps lead
  (SPY >> RSP, RSP failing 100d SMA), QQQ's tech concentration adds risk
  without proportional reward.
- RSP 100d SMA as trend filter (faster than 200d): catches breadth breaks
  earlier than standard 200d trend filters.
- Vol hedge overlay: during vol spikes (above 80th pct of 90d RV), any equity
  is at elevated risk — reduce exposure temporarily. This is an OVERLAY not
  a primary signal (unlike gen7 realized vol carry which used vol carry as
  the PRIMARY sizing signal).
- This combination — QQQ + RSP breadth gate + vol overlay — is distinct from
  all existing leaderboard strategies.

Differentiators:
- RSP 100d SMA as QQQ breadth qualification (not binary JNK/VIX gate)
- QQQ → SPY downgrade when breadth fails (not QQQ → TLT binary)
- Vol hedge as secondary overlay (not primary regime gate)
- Weekly rebalance for responsiveness
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5        # weekly
RSP_TREND_WINDOW = 100     # RSP breadth trend gate (fast)
SPY_TREND_WINDOW = 200     # SPY outer bear gate
RV_WINDOW = 21             # realized vol window
RV_PERCENTILE_WINDOW = 90  # percentile reference window
RV_STRESSED_PCT = 80       # above 80th pct: vol stressed
# Base exposures
QQQ_FULL = 0.97            # QQQ when RSP breadth + SPY bull
SPY_MID = 0.97             # SPY when RSP breadth fails but SPY bull
EXPOSURE_STRESSED = 0.65   # reduce when vol stressed
TLT_STRESSED = 0.32        # TLT fill when vol stressed

_SPY = "SPY"
_QQQ = "QQQ"
_TLT = "TLT"
_RSP = "RSP"


class QqqRspBreadthVolHedge(Strategy):
    """QQQ with RSP 100d breadth gate and vol overlay: QQQ 97% / SPY 97% / TLT."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        rsp_trend_window: int = RSP_TREND_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        rv_window: int = RV_WINDOW,
        rv_percentile_window: int = RV_PERCENTILE_WINDOW,
        rv_stressed_pct: float = RV_STRESSED_PCT,
        qqq_full: float = QQQ_FULL,
        spy_mid: float = SPY_MID,
        exposure_stressed: float = EXPOSURE_STRESSED,
        tlt_stressed: float = TLT_STRESSED,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            rsp_trend_window=rsp_trend_window,
            spy_trend_window=spy_trend_window,
            rv_window=rv_window,
            rv_percentile_window=rv_percentile_window,
            rv_stressed_pct=rv_stressed_pct,
            qqq_full=qqq_full,
            spy_mid=spy_mid,
            exposure_stressed=exposure_stressed,
            tlt_stressed=tlt_stressed,
        )
        self.rebalance_every = int(rebalance_every)
        self.rsp_trend_window = int(rsp_trend_window)
        self.spy_trend_window = int(spy_trend_window)
        self.rv_window = int(rv_window)
        self.rv_percentile_window = int(rv_percentile_window)
        self.rv_stressed_pct = float(rv_stressed_pct)
        self.qqq_full = float(qqq_full)
        self.spy_mid = float(spy_mid)
        self.exposure_stressed = float(exposure_stressed)
        self.tlt_stressed = float(tlt_stressed)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend_window,
                     self.rv_percentile_window + self.rv_window) + 10
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
                target[_TLT] = self.qqq_full
        else:
            # --- RSP 100d breadth gate ---
            rsp_broad = True  # default: broad breadth
            try:
                rsp_hist = ctx.history(_RSP)
                rsp_c = rsp_hist["close"].dropna()
                if len(rsp_c) >= self.rsp_trend_window + 2:
                    rsp_sma = float(rsp_c.iloc[-self.rsp_trend_window:].mean())
                    rsp_now = float(rsp_c.iloc[-1])
                    rsp_broad = rsp_now > rsp_sma
            except Exception:
                pass

            # --- SPY realized vol overlay ---
            vol_stressed = False
            try:
                if len(spy_close) >= self.rv_percentile_window + self.rv_window + 5:
                    log_rets = np.log(spy_close.values[1:] / spy_close.values[:-1])
                    current_rv = float(np.std(log_rets[-self.rv_window:]) * np.sqrt(252))
                    rv_series = []
                    for i in range(self.rv_percentile_window):
                        end_i = len(log_rets) - i
                        start_i = end_i - self.rv_window
                        if start_i < 0:
                            break
                        rv_series.append(float(np.std(log_rets[start_i:end_i]) * np.sqrt(252)))
                    if rv_series and np.isfinite(current_rv):
                        stressed_thr = float(np.percentile(rv_series, self.rv_stressed_pct))
                        vol_stressed = current_rv >= stressed_thr
            except Exception:
                pass

            # --- Route: QQQ vs SPY based on breadth ---
            primary_sym = _QQQ if rsp_broad else _SPY
            primary_exp = self.qqq_full if rsp_broad else self.spy_mid

            if vol_stressed:
                # Reduce equity, add TLT
                if primary_sym in live:
                    target[primary_sym] = self.exposure_stressed
                if _TLT in live:
                    target[_TLT] = self.tlt_stressed
            else:
                if primary_sym in live:
                    target[primary_sym] = primary_exp

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


UNIVERSE = [_SPY, _QQQ, _TLT, _RSP]

NAME = "jnk_gated_sector_momentum"
HYPOTHESIS = (
    "QQQ with RSP 100d breadth gate + vol overlay: QQQ 97% when RSP above 100d SMA "
    "(broad breadth) AND SPY bull; SPY 97% when RSP breadth fails; reduce to 65% + "
    "TLT 32% when vol stressed (SPY 21d RV above 90d 80th pct); TLT when SPY bear; "
    "weekly rebalance."
)

STRATEGY = QqqRspBreadthVolHedge()
