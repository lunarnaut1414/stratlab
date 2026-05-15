"""gen_9 opus-1 — Idiosyncratic Momentum + Short-End Yield Slope Gate.

Parent: gen9_idio_momentum_longend_slope (IS Calmar 0.97).
Mutation:
  - Slope segment: TYX-TNX (30Y-10Y long-end) -> TNX-IRX (10Y-3M short-end)
  - Same idiosyncratic 63d momentum (alpha = raw_return - beta * spy_return)
  - Same SPY 200d outer bear gate -> TLT
  - Same inverse-vol weighting of top-15
  - Same risk-off fallback SPY 60% + IEF 37% when slope flat/negative

Rationale: The short-end slope (TNX-IRX) captures Fed-policy regime (steepening
= dovish/easing cycle, flattening = hawkish/tightening). The long-end slope
captures duration/inflation risk premium. They are distinct macro signals that
fire at different times. The Phase 2 brief notes "different segment = different
timing" — that's precisely the diversification angle (different daily PnL path
even when both gate the same stock selector).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
MOMENTUM_WINDOW = 63
BETA_WINDOW = 126
SLOPE_MA_WINDOW = 200
SPY_TREND_WINDOW = 200
VOL_WINDOW = 21
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_TNX = "^TNX"
_IRX = "^IRX"


class Opus1IdioMomShortendSlope(Strategy):
    """SP500 idiosyncratic momentum gated by short-end yield slope (TNX-IRX)."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(BETA_WINDOW, SLOPE_MA_WINDOW, SPY_TREND_WINDOW) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_EVERY != 0:
            return []

        # SPY 200d outer bear gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND_WINDOW + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < SPY_TREND_WINDOW:
            return []
        spy_sma = float(spy_close.iloc[-SPY_TREND_WINDOW:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        # Short-end TNX-IRX slope vs 200d MA
        slope_steep = True
        try:
            tnx_hist = ctx.history(_TNX)
            irx_hist = ctx.history(_IRX)
            if (tnx_hist is not None and irx_hist is not None
                    and len(tnx_hist) >= SLOPE_MA_WINDOW + 2
                    and len(irx_hist) >= SLOPE_MA_WINDOW + 2):
                tnx_c = tnx_hist["close"].dropna()
                irx_c = irx_hist["close"].dropna()
                n = min(len(tnx_c), len(irx_c))
                if n >= SLOPE_MA_WINDOW + 1:
                    slope = tnx_c.values[-n:] - irx_c.values[-n:]
                    slope_ma = float(np.mean(slope[-SLOPE_MA_WINDOW:]))
                    slope_now = float(slope[-1])
                    slope_steep = slope_now > slope_ma
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
            if _TLT in live:
                target[_TLT] = EXPOSURE
        elif not slope_steep:
            if _SPY in live:
                target[_SPY] = EXPOSURE * 0.618
            if _IEF in live:
                target[_IEF] = EXPOSURE * 0.382
        else:
            need = max(BETA_WINDOW, MOMENTUM_WINDOW) + 5
            prices = ctx.closes_window(need)
            if len(prices) < MOMENTUM_WINDOW + 5:
                if _SPY in live:
                    target[_SPY] = EXPOSURE
            else:
                if _SPY not in prices.columns:
                    if _SPY in live:
                        target[_SPY] = EXPOSURE
                else:
                    spy_prices = prices[_SPY].dropna()
                    if len(spy_prices) < BETA_WINDOW:
                        if _SPY in live:
                            target[_SPY] = EXPOSURE
                    else:
                        spy_log_rets = np.log(
                            spy_prices.values[1:] / spy_prices.values[:-1]
                        )
                        spy_mom_ret = float(
                            spy_prices.iloc[-1] / spy_prices.iloc[-MOMENTUM_WINDOW] - 1.0
                        )

                        scores: dict[str, float] = {}
                        vols: dict[str, float] = {}

                        for sym in prices.columns:
                            if sym in (_SPY, _TLT, _IEF):
                                continue
                            col = prices[sym].dropna()
                            if len(col) < BETA_WINDOW:
                                continue

                            stock_log_rets = np.log(col.values[1:] / col.values[:-1])
                            n = min(len(stock_log_rets), len(spy_log_rets))
                            if n < 30:
                                continue
                            sr = stock_log_rets[-n:]
                            mr = spy_log_rets[-n:]
                            var_m = np.var(mr)
                            if var_m < 1e-10:
                                continue
                            beta = float(np.cov(sr, mr)[0, 1] / var_m)
                            if not np.isfinite(beta):
                                continue

                            if len(col) < MOMENTUM_WINDOW + 1:
                                continue
                            raw_ret = float(
                                col.iloc[-1] / col.iloc[-MOMENTUM_WINDOW] - 1.0
                            )
                            if not np.isfinite(raw_ret):
                                continue

                            idio = raw_ret - beta * spy_mom_ret
                            if not np.isfinite(idio):
                                continue
                            scores[sym] = idio

                            vol_rets = stock_log_rets[-min(VOL_WINDOW, len(stock_log_rets)):]
                            rv = float(np.std(vol_rets)) * np.sqrt(252)
                            vols[sym] = rv if rv > 1e-6 else 1e-6

                        if len(scores) < 5:
                            if _TLT in live:
                                target[_TLT] = EXPOSURE
                        else:
                            k = min(TOP_K, len(scores))
                            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                            inv_vols = {sym: 1.0 / vols.get(sym, 1.0) for sym in ranked}
                            total_iv = sum(inv_vols.values())
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = EXPOSURE * inv_vols[sym] / total_iv

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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _TNX, _IRX]


UNIVERSE = _universe

NAME = "opus1_idio_mom_shortend_slope"
HYPOTHESIS = (
    "Mutate gen9_idio_momentum_longend_slope: swap long-end slope (TYX-TNX) "
    "for short-end slope (TNX-IRX) vs 200d MA; same idiosyncratic 63d momentum, "
    "SPY 200d bear gate -> TLT, risk-off SPY+IEF blend when slope flat; "
    "short-end captures Fed-policy regime, different timing than long-end."
)

STRATEGY = Opus1IdioMomShortendSlope()
