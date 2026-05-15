"""EM vs US regime gate driving SP500 momentum vs defensive allocation.

Hypothesis: Use the 60-day return spread between EEM (emerging markets ETF) and
SPY (US large cap) as a global risk appetite and USD-strength signal. When
emerging markets outperform US equities, it signals risk-on globally, often
coinciding with USD weakness — a historically favorable environment for US
growth stocks. When US equities lead EM, it signals relative USD strength and
often a more defensive equity regime.

Signal tiers:
  1. EEM 60d return > SPY 60d return + 2pp (global risk-on, EM leading):
     Hold top-10 SP500 stocks by 63d momentum (risk-seeking).
  2. SPY 60d return > EEM 60d return (US leading, USD strength):
     Hold SPY 60% + IEF 37% (modest defensive tilt).
  3. Both EEM and SPY have negative 60d returns (risk-off globally):
     Hold TLT 97% (full defensive).
  4. Override: SPY below 200d SMA → always TLT.

Rationale: The VEA/VWO signal from gen7 had 0.74 OOS Calmar with low corr.
This strategy uses EEM vs SPY (EM vs US developed) rather than VWO vs VEA
(EM vs DM international) — a different relative signal that captures USD cycles
more directly. EEM and SPY are both signal-only inputs routed to US assets only.

Distinction from leaderboard:
  - gen7_opus2_vea_vwo_signal_us_only: uses VEA/VWO ratio (DM vs EM international),
    this uses EEM vs SPY (EM vs US, capturing USD strength cycle directly).
  - All other entries use VIX, credit spreads, yield curve, or dollar ETF.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly
MOM_WINDOW = 63             # SP500 stock momentum window
RATIO_WINDOW = 60           # EM vs US comparison window
EM_LEAD_THRESHOLD = 0.02    # EEM must beat SPY by >2pp to signal risk-on
SPY_TREND_WINDOW = 200
TOP_K = 10
EXPOSURE = 0.97


class EmUsRegimeGate(Strategy):
    """EM vs US 60d return spread gate: SP500 momentum (EM leads) or SPY+IEF
    (US leads) or TLT (both negative); SPY 200d outer bear gate.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        ratio_window: int = RATIO_WINDOW,
        em_lead_threshold: float = EM_LEAD_THRESHOLD,
        spy_trend_window: int = SPY_TREND_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            ratio_window=ratio_window,
            em_lead_threshold=em_lead_threshold,
            spy_trend_window=spy_trend_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.ratio_window = int(ratio_window)
        self.em_lead_threshold = float(em_lead_threshold)
        self.spy_trend_window = int(spy_trend_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_window, self.ratio_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA outer gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma200 = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma200

        # EEM signal (signal-only — not traded)
        try:
            eem_hist = ctx.history("EEM")
        except KeyError:
            eem_hist = None

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market override → TLT
            if "TLT" in closes_now.index:
                target["TLT"] = self.exposure
        else:
            # Compute EEM and SPY 60d returns for regime classification
            eem_ret = float("nan")
            spy_ret = float("nan")

            if eem_hist is not None and len(eem_hist) >= self.ratio_window + 2:
                eem_c = eem_hist["close"].dropna()
                if len(eem_c) >= self.ratio_window + 1:
                    eem_ret = float(eem_c.iloc[-1] / eem_c.iloc[-self.ratio_window] - 1.0)

            if len(spy_close) >= self.ratio_window + 1:
                spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.ratio_window] - 1.0)

            if not np.isfinite(eem_ret) or not np.isfinite(spy_ret):
                # Fallback: SPY+IEF blend
                if "SPY" in closes_now.index and "IEF" in closes_now.index:
                    target["SPY"] = 0.60 * self.exposure
                    target["IEF"] = 0.37 * self.exposure
            elif spy_ret < 0 and eem_ret < 0:
                # Both negative: full defensive TLT
                if "TLT" in closes_now.index:
                    target["TLT"] = self.exposure
            elif eem_ret > spy_ret + self.em_lead_threshold:
                # EM leads by >2pp: global risk-on → top-K SP500 momentum
                prices = ctx.closes_window(self.mom_window + 5)
                if len(prices) < self.mom_window:
                    if "SPY" in closes_now.index:
                        target["SPY"] = self.exposure
                else:
                    scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in ("EEM", "SPY", "IEF", "TLT"):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.mom_window:
                            continue
                        ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                        if np.isfinite(ret):
                            scores[sym] = ret

                    if len(scores) < self.top_k:
                        if "SPY" in closes_now.index:
                            target["SPY"] = self.exposure
                    else:
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                        longs = ranked[:self.top_k]
                        per_weight = self.exposure / len(longs)
                        for sym in longs:
                            target[sym] = per_weight
            else:
                # US leads EM (USD strength regime): SPY 60% + IEF 37%
                if "SPY" in closes_now.index and "IEF" in closes_now.index:
                    target["SPY"] = 0.60 * self.exposure
                    target["IEF"] = 0.37 * self.exposure

        # Build orders
        orders: list[Order] = []

        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Size to target
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
    # EEM is signal-only (non-US); include SPY and bonds as holdings
    return sp500_tickers() + ["TLT", "IEF", "SPY", "EEM"]


NAME = "em_us_regime_gate"
HYPOTHESIS = (
    "EM vs US regime gate: use EEM 60d return vs SPY 60d return spread as global risk appetite "
    "signal; when EEM leads SPY by >2pp (global risk-on, USD weak) hold top-10 SP500 stocks by "
    "63d momentum; when SPY leads EEM (USD strong) hold SPY 60%+IEF 37%; when both below 0 hold "
    "TLT; SPY 200d outer bear gate; biweekly rebalance — EM-vs-US flow signal not present in leaderboard"
)

UNIVERSE = _universe

STRATEGY = EmUsRegimeGate()
