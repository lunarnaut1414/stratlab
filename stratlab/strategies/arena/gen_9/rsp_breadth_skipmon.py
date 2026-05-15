"""gen_9 sonnet-3 — RSP/SPY Breadth-Gated Skip-Month SP500 Momentum

Hypothesis: RSP (equal-weight SP500) vs SPY (cap-weight SP500) 21d return spread
as a market-breadth regime gate on skip-month momentum stock selection.

When RSP outperforms SPY on 21d basis (equal-weight outperforming):
  -> Broad market participation, risk-on: hold top-20 skip-month (126d-skip-21d)
     SP500 stocks above their 50d SMA, inverse-vol weighted
     (skip-month momentum = proven OOS gen8, avoids short-term reversal)

When RSP underperforms SPY or neutral:
  -> Narrow/mega-cap leadership (lower breadth): hold SPY 97%
     (concentrated leadership regimes favor the index itself)

When SPY below 200d SMA:
  -> Bear market: hold TLT 97%

Rationale:
  - Skip-month (126d-skip-21d) is OOS-robust (gen8 OOS Calmar 0.63, 79% IS retention)
  - RSP vs SPY breadth is a novel gate NOT used in combination with skip-month
  - RSP leadership = market breadth expanding (2010, 2013-2014, 2016 fired ~50-68%)
  - SPY leadership = mega-cap concentration (2015, 2017, 2018 fired only 31-35%)
    In mega-cap-led regimes, stock selection is harder; SPY is safer

Distinction from existing:
  - gen8_sp500_skipmon_63sma_momentum: uses SPY 200d SMA market gate (not RSP breadth)
    and stock 63d SMA (not 50d), no RSP signal
  - gen8_rsp_spy_breadth_qqq_rotation (IS 0.79, OOS 0.17 severe overfit): routes to QQQ
    vs SPY+IEF (ETF routing not stock selection), used 42d not 21d, DIFFERENT goal
  - This strategy uses RSP breadth to GATE individual SP500 stock selection
    vs just routing between ETFs

Why different timing from existing:
  - RSP outperforms 49% of IS time (real differentiation)
  - Combines RSP-breadth (timing) with skip-month (quality of selection)
  - Neither leg has been tried in combination before
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOM_LONG = 126            # skip-month long window
MOM_SKIP = 21             # skip most recent 21 days
BREADTH_WINDOW = 21       # RSP vs SPY return comparison window
SPY_TREND = 200           # SPY 200d SMA bear gate
STOCK_SMA = 50            # per-stock 50d SMA filter
VOL_WINDOW = 21
TOP_K = 20
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_RSP = "RSP"


class RspBreadthSkipmon(Strategy):
    """RSP/SPY 21d breadth gates skip-month SP500 momentum stock selection."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_long: int = MOM_LONG,
        mom_skip: int = MOM_SKIP,
        breadth_window: int = BREADTH_WINDOW,
        spy_trend: int = SPY_TREND,
        stock_sma: int = STOCK_SMA,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_long=mom_long,
            mom_skip=mom_skip,
            breadth_window=breadth_window,
            spy_trend=spy_trend,
            stock_sma=stock_sma,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_long = int(mom_long)
        self.mom_skip = int(mom_skip)
        self.breadth_window = int(breadth_window)
        self.spy_trend = int(spy_trend)
        self.stock_sma = int(stock_sma)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.spy_trend, self.mom_long, self.stock_sma) + 10
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
            if spy_hist is not None and len(spy_hist) >= self.spy_trend + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.spy_trend:
                    spy_sma = float(spy_close.iloc[-self.spy_trend:].mean())
                    spy_now = float(spy_close.iloc[-1])
                    spy_bull = spy_now > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # --- RSP vs SPY 21d breadth signal ---
            rsp_leads = False
            try:
                rsp_hist = ctx.history(_RSP)
                spy_hist = ctx.history(_SPY)
                if (rsp_hist is not None and spy_hist is not None and
                        len(rsp_hist) >= self.breadth_window + 2 and
                        len(spy_hist) >= self.breadth_window + 2):
                    rsp_close = rsp_hist["close"].dropna()
                    spy_close_b = spy_hist["close"].dropna()
                    if (len(rsp_close) >= self.breadth_window + 1 and
                            len(spy_close_b) >= self.breadth_window + 1):
                        rsp_ret = float(
                            rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0
                        )
                        spy_ret_b = float(
                            spy_close_b.iloc[-1] / spy_close_b.iloc[-self.breadth_window] - 1.0
                        )
                        rsp_leads = np.isfinite(rsp_ret) and np.isfinite(spy_ret_b) and rsp_ret > spy_ret_b
            except Exception:
                pass

            if not rsp_leads:
                # Narrow / mega-cap leadership: hold SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Broad breadth: skip-month momentum stock selection
                need = self.mom_long + 10
                prices = ctx.closes_window(need)
                if len(prices) < self.mom_long:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    # Skip-month: 126d-skip-21d return
                    scores: dict[str, float] = {}
                    for sym in prices.columns:
                        if sym in (_SPY, _TLT, _RSP):
                            continue
                        col = prices[sym].dropna()
                        if len(col) < self.mom_long + 5:
                            continue

                        p_skip = float(col.iloc[-(self.mom_skip + 1)])
                        if len(col) > self.mom_long:
                            p_start = float(col.iloc[-(self.mom_long + 1)])
                        else:
                            p_start = float(col.iloc[0])
                        if p_start <= 0 or p_skip <= 0:
                            continue
                        ret = p_skip / p_start - 1.0
                        if np.isfinite(ret):
                            scores[sym] = ret

                    if len(scores) < 5:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                        # Per-stock 50d SMA filter + inverse-vol weighting
                        inv_vols: dict[str, float] = {}
                        count = 0
                        for sym in ranked:
                            if count >= self.top_k:
                                break
                            try:
                                s_hist = ctx.history(sym)
                                if s_hist is None:
                                    continue
                                s_close = s_hist["close"].dropna()
                                if len(s_close) < self.stock_sma + 2:
                                    continue
                                sma_val = float(s_close.iloc[-self.stock_sma:].mean())
                                p_now = float(s_close.iloc[-1])
                                if p_now <= sma_val:
                                    continue  # below 50d SMA

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
                                count += 1

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
    return sp500_tickers() + [_TLT, _SPY, _RSP]


UNIVERSE = _universe

NAME = "rsp_breadth_skipmon"
HYPOTHESIS = (
    "RSP vs SPY 21d breadth signal gates skip-month SP500 momentum: "
    "RSP outperforms SPY on 21d (broad breadth, risk-on) -> top-20 skip-month (126d-skip-21d) "
    "SP500 stocks with per-stock 50d SMA filter inverse-vol weighted; "
    "RSP lags SPY or neutral -> SPY 97%; SPY below 200d SMA -> TLT; "
    "combines proven skip-month momentum with novel RSP-breadth gate distinct from existing "
    "SPY/JNK/TNX regime signals"
)

STRATEGY = RspBreadthSkipmon()
