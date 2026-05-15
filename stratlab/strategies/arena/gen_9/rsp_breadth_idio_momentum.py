"""gen_9 sonnet-6 — RSP/SPY Breadth Quality Gate with Idiosyncratic Momentum

Hypothesis: When RSP (equal-weight SP500) 60d return exceeds SPY 60d return
(broad participation, breadth expanding), market rally has quality behind it.
Hold top-15 SP500 stocks by idiosyncratic momentum (63d residual vs SPY beta).
When SPY leads RSP (narrow mega-cap driven rally), reduce confidence: hold
SPY 60%+IEF 37%. SPY 200d outer bear gate to TLT.

Rationale: RSP>SPY means the average stock is participating — a structural
signal of rally health distinct from VIX, credit spreads, or yield curves.
Combining this with idiosyncratic stock momentum (residual alpha vs SPY)
targets stocks that outperform the market on a quality-adjusted basis only
when broad breadth confirms the rally. The two filters compound: narrow market
AND stock alpha = double confirmation of genuine momentum.

RSP covers IS from 2003-05-01 — full IS coverage (2010-2018).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
BETA_WINDOW = 126
TREND_WINDOW = 200
BREADTH_WINDOW = 60    # 60d for RSP vs SPY comparison
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_RSP = "RSP"


class RspBreadthIdioMomentum(Strategy):
    """SP500 idiosyncratic momentum gated by RSP vs SPY breadth quality."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        trend_window: int = TREND_WINDOW,
        breadth_window: int = BREADTH_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            beta_window=beta_window,
            trend_window=trend_window,
            breadth_window=breadth_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.trend_window = int(trend_window)
        self.breadth_window = int(breadth_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.beta_window, self.breadth_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d outer bear gate ---
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # --- RSP vs SPY breadth gate ---
        breadth_on = True  # default risk-on if signal unavailable
        try:
            rsp_hist = ctx.history(_RSP)
            if rsp_hist is not None and len(rsp_hist) >= self.breadth_window + 2:
                rsp_close = rsp_hist["close"].dropna()
                if len(rsp_close) >= self.breadth_window + 1:
                    rsp_ret = float(rsp_close.iloc[-1] / rsp_close.iloc[-self.breadth_window] - 1.0)
                    spy_ret_60 = float(spy_close.iloc[-1] / spy_close.iloc[-self.breadth_window] - 1.0)
                    # RSP > SPY = broad participation = risk-on
                    breadth_on = np.isfinite(rsp_ret) and np.isfinite(spy_ret_60) and rsp_ret >= spy_ret_60
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
            # Bear regime: full TLT defensive
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not breadth_on:
            # SPY leads RSP — narrow market: SPY+IEF blend
            if _SPY in live:
                target[_SPY] = self.exposure * 0.62
            if _IEF in live:
                target[_IEF] = self.exposure * 0.35
        else:
            # Broad breadth + bull: top-K idiosyncratic momentum stocks
            need = max(self.beta_window, self.momentum_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                if _SPY not in prices.columns:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    spy_prices = prices[_SPY].dropna()
                    if len(spy_prices) < self.beta_window:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        spy_log_rets = np.log(spy_prices.values[1:] / spy_prices.values[:-1])
                        spy_mom_ret = float(
                            spy_prices.iloc[-1] / spy_prices.iloc[-self.momentum_window] - 1.0
                        )

                        scores: dict[str, float] = {}
                        for sym in prices.columns:
                            if sym in (_SPY, _TLT, _IEF, _RSP):
                                continue
                            col = prices[sym].dropna()
                            if len(col) < self.beta_window:
                                continue
                            stock_log_rets = np.log(col.values[1:] / col.values[:-1])
                            n = min(len(stock_log_rets), len(spy_log_rets))
                            if n < 30:
                                continue
                            stock_r = stock_log_rets[-n:]
                            spy_r = spy_log_rets[-n:]
                            spy_var = float(np.var(spy_r))
                            if spy_var < 1e-8:
                                continue
                            beta = float(np.cov(stock_r, spy_r)[0, 1] / spy_var)
                            if not np.isfinite(beta):
                                continue
                            if len(col) < self.momentum_window + 1:
                                continue
                            raw_ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                            if not np.isfinite(raw_ret):
                                continue
                            idio_ret = raw_ret - beta * spy_mom_ret
                            if np.isfinite(idio_ret):
                                scores[sym] = idio_ret

                        if len(scores) < 5:
                            if _SPY in live:
                                target[_SPY] = self.exposure
                        else:
                            k = min(self.top_k, len(scores))
                            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                            per_weight = self.exposure / len(ranked)
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = per_weight

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
    return sp500_tickers() + [_TLT, _SPY, _IEF, _RSP]


NAME = "rsp_breadth_idio_momentum"
HYPOTHESIS = (
    "RSP vs SPY breadth quality gate: when RSP (equal-weight SP500) 60d return exceeds SPY "
    "60d return (broad participation, breadth expanding), hold top-15 SP500 stocks by 63d "
    "idiosyncratic momentum (residual vs SPY) above 200d SMA; when SPY leads RSP (narrow "
    "mega-cap driven), hold SPY 60%+IEF 37%; SPY 200d outer bear gate to TLT; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = RspBreadthIdioMomentum()
