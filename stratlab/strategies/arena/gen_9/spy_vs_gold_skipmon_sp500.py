"""gen_9 sonnet-1 — SPY/Gold 42d Return Signal → Skip-Month SP500 Momentum

Hypothesis: Use the relative performance of SPY vs gold (IAU) on a 42d basis
as an equity-vs-safe-haven regime signal.
- SPY 42d return > IAU 42d return (equities leading gold = genuine risk-on) →
  hold top-15 SP500 stocks by 126d skip-21d (skip-month) momentum, above 63d SMA,
  inverse-vol weighted; SPY 200d outer bear gate overrides to TLT.
- IAU 42d return > SPY 42d return (gold leading equities = risk-off / inflation fear) →
  SPY 60% + TLT 37% (moderate risk-off blend).

Rationale:
- SPY/gold relative performance captures a DIFFERENT regime dimension than VIX
  (fear gauge), JNK (credit spread), or TNX (yield level). Gold outperforms SPY
  in multiple distinct conditions: (a) early-stage risk-off before VIX spikes,
  (b) genuine inflation concerns, (c) tail-risk events. This creates different
  entry/exit timing.
- Skip-month SP500 momentum (126d-21d) avoids 1-month reversal contamination —
  OOS-robust per gen_8 results.
- Per-stock 63d SMA filter provides intra-stock trend confirmation.
- Inverse-vol weighting prevents high-volatility winners dominating.
- Never combined before: gold/equity RATIO signal routing to skip-month stock selection.

Coverage (all cover IS 2010-2018):
  IAU (2005), TLT (2002), SPY (1993), SP500 stocks
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

SIGNAL_WINDOW = 42      # SPY vs IAU 42d return comparison
MOM_LONG = 126          # skip-month: 126d total return
MOM_SKIP = 21           # skip-month: skip last 21d
STOCK_SMA = 63          # per-stock 63d SMA filter
SPY_TREND = 200         # 200d SMA outer bear gate
VOL_WINDOW = 21         # 21d realized vol for inverse-vol weighting
TOP_K = 15
EXPOSURE = 0.97
REBALANCE_EVERY = 10    # biweekly

_IAU = "IAU"
_TLT = "TLT"
_SPY = "SPY"


class SpyVsGoldSkipMonSp500(Strategy):
    """SPY/gold 42d return spread gates skip-month SP500 momentum."""

    def __init__(
        self,
        signal_window: int = SIGNAL_WINDOW,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        stock_sma: int = STOCK_SMA,
        spy_trend: int = SPY_TREND,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        rebalance_every: int = REBALANCE_EVERY,
    ) -> None:
        super().__init__(
            signal_window=signal_window,
            mom_long=mom_long,
            mom_skip=mom_skip,
            stock_sma=stock_sma,
            spy_trend=spy_trend,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
            rebalance_every=rebalance_every,
        )
        self.signal_window = int(signal_window)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.stock_sma = int(stock_sma)
        self.spy_trend = int(spy_trend)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.rebalance_every = int(rebalance_every)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_long + 5, self.spy_trend + 5, self.signal_window + 5) + 10
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

        # --- SPY vs IAU regime signal ---
        equity_leads: bool | None = None
        try:
            iau_hist = ctx.history(_IAU)
            if iau_hist is not None and len(iau_hist) >= self.signal_window + 2:
                iau_c = iau_hist["close"].dropna()
                if len(iau_c) >= self.signal_window and len(spy_close) >= self.signal_window:
                    iau_ret = float(iau_c.iloc[-1] / iau_c.iloc[-self.signal_window] - 1.0)
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-self.signal_window] - 1.0)
                    if np.isfinite(iau_ret) and np.isfinite(spy_ret):
                        equity_leads = spy_ret > iau_ret
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

        if not spy_bull:
            # Bear market → TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif equity_leads is False:
            # Gold leading equities → moderate risk-off
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _TLT in live:
                target[_TLT] = self.exposure * 0.382
        else:
            # SPY leads gold (or signal unavailable) → skip-month SP500 momentum
            need = self.mom_long + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_long:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                scores: dict[str, float] = {}
                vols: dict[str, float] = {}

                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IAU):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_long:
                        continue

                    # Skip-month momentum: 126d return excluding last 21d
                    if len(col) < self.mom_skip + 1:
                        continue
                    p_skip = float(col.iloc[-self.mom_skip])    # price 21d ago
                    p_start = float(col.iloc[-self.mom_long])   # price 126d ago
                    if p_start <= 0:
                        continue
                    skip_ret = float(p_skip / p_start - 1.0)
                    if not np.isfinite(skip_ret):
                        continue

                    # Per-stock 63d SMA filter
                    if len(col) < self.stock_sma:
                        continue
                    sma_val = float(col.iloc[-self.stock_sma:].mean())
                    curr_price = float(col.iloc[-1])
                    if curr_price <= sma_val:
                        continue

                    scores[sym] = skip_ret

                    # Realized vol for inverse-vol weighting
                    log_rets = np.log(col.values[1:] / col.values[:-1])
                    rv_slice = log_rets[-min(self.vol_window, len(log_rets)):]
                    rv = float(np.std(rv_slice)) * np.sqrt(252)
                    vols[sym] = rv if rv > 1e-6 else 1e-6

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    k = min(self.top_k, len(scores))
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                    inv_vols = {sym: 1.0 / vols.get(sym, 1.0) for sym in ranked}
                    total_iv = sum(inv_vols.values())
                    for sym in ranked:
                        if sym in live:
                            target[sym] = self.exposure * inv_vols[sym] / total_iv

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
    return sp500_tickers() + [_TLT, _IAU, _SPY]


NAME = "spy_vs_gold_skipmon_sp500"
HYPOTHESIS = (
    "SPY vs IAU gold 42d return as equity/safe-haven regime signal: "
    "SPY leads gold → top-15 SP500 by 126d skip-21d momentum above 63d SMA, "
    "inverse-vol weighted; gold leads SPY → SPY 62%+TLT 38%; "
    "SPY 200d bear → TLT; biweekly rebalance; "
    "SPY/gold spread routing to skip-month stock selection is novel combination"
)

UNIVERSE = _universe

STRATEGY = SpyVsGoldSkipMonSp500()
