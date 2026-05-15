"""gen_9 sonnet-1 — Idiosyncratic Momentum + Long-End Yield Slope Gate

Hypothesis: Rank SP500 stocks by idiosyncratic 63d return (raw return minus
beta × SPY return), gated by the long-end yield-curve slope (TYX-TNX) vs its
own 200d MA.

Rationale:
- The idiosyncratic momentum signal (gen_7 best OOS: 0.70) selects stocks that
  are beating the market on a risk-adjusted basis, not just riding beta.
- The long-end slope gate (gen_8 best OOS: 0.79) proved the most OOS-stable
  macro regime signal in the leaderboard.
- Combining the best OOS stock signal with the best OOS macro gate should
  produce both a high IS Calmar and better OOS retention than either parent.

Regime logic:
  - SPY below 200d SMA → fully defensive TLT (outer bear gate)
  - SPY above 200d SMA AND TYX-TNX > its 200d MA (slope steep/rising) →
      top-15 SP500 by idiosyncratic alpha, inverse-vol weighted
  - SPY above 200d SMA AND TYX-TNX < its 200d MA (slope flat/falling) →
      SPY 60% + IEF 37% (moderate risk-off blend)
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10        # biweekly (~2 weeks)
MOMENTUM_WINDOW = 63        # 3-month momentum
BETA_WINDOW = 126           # 6-month beta estimation
SLOPE_MA_WINDOW = 200       # 200d MA on the TYX-TNX slope
SPY_TREND_WINDOW = 200      # 200d SMA bear gate
VOL_WINDOW = 21             # 21d realized vol for inverse-vol weighting
TOP_K = 15
EXPOSURE = 0.97

_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"
_TYX = "^TYX"
_TNX = "^TNX"


class IdioMomentumLongEndSlope(Strategy):
    """SP500 idiosyncratic momentum gated by 30Y-10Y long-end yield slope."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        beta_window: int = BETA_WINDOW,
        slope_ma_window: int = SLOPE_MA_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            beta_window=beta_window,
            slope_ma_window=slope_ma_window,
            spy_trend_window=spy_trend_window,
            vol_window=vol_window,
            top_k=top_k,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.beta_window = int(beta_window)
        self.slope_ma_window = int(slope_ma_window)
        self.spy_trend_window = int(spy_trend_window)
        self.vol_window = int(vol_window)
        self.top_k = int(top_k)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.beta_window, self.slope_ma_window, self.spy_trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY 200d SMA outer bear gate ---
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

        # --- TYX-TNX long-end slope vs 200d MA ---
        slope_steep = True  # default risk-on if signal unavailable
        try:
            tyx_hist = ctx.history(_TYX)
            tnx_hist = ctx.history(_TNX)
            if (tyx_hist is not None and tnx_hist is not None
                    and len(tyx_hist) >= self.slope_ma_window + 2
                    and len(tnx_hist) >= self.slope_ma_window + 2):
                tyx_c = tyx_hist["close"].dropna()
                tnx_c = tnx_hist["close"].dropna()
                n = min(len(tyx_c), len(tnx_c))
                if n >= self.slope_ma_window + 1:
                    slope = tyx_c.values[-n:] - tnx_c.values[-n:]
                    slope_ma = float(np.mean(slope[-self.slope_ma_window:]))
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
            # Full bear → TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        elif not slope_steep:
            # Long-end slope flat/falling → moderate risk-off
            if _SPY in live:
                target[_SPY] = self.exposure * 0.618
            if _IEF in live:
                target[_IEF] = self.exposure * 0.382
        else:
            # Bull + steep long-end slope → idiosyncratic momentum
            need = max(self.beta_window, self.momentum_window) + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window + 5:
                # Not enough history yet → fallback SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            else:
                # Compute SPY returns for beta
                if _SPY not in prices.columns:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    spy_prices = prices[_SPY].dropna()
                    if len(spy_prices) < self.beta_window:
                        if _SPY in live:
                            target[_SPY] = self.exposure
                    else:
                        spy_log_rets = np.log(
                            spy_prices.values[1:] / spy_prices.values[:-1]
                        )
                        spy_mom_ret = float(
                            spy_prices.iloc[-1] / spy_prices.iloc[-self.momentum_window] - 1.0
                        )

                        scores: dict[str, float] = {}
                        vols: dict[str, float] = {}

                        for sym in prices.columns:
                            if sym in (_SPY, _TLT, _IEF):
                                continue
                            col = prices[sym].dropna()
                            if len(col) < self.beta_window:
                                continue

                            # Beta via covariance
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

                            # 63d raw momentum
                            if len(col) < self.momentum_window + 1:
                                continue
                            raw_ret = float(
                                col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0
                            )
                            if not np.isfinite(raw_ret):
                                continue

                            # Idiosyncratic return
                            idio = raw_ret - beta * spy_mom_ret
                            if not np.isfinite(idio):
                                continue
                            scores[sym] = idio

                            # Realized vol for inverse-vol weighting
                            vol_rets = stock_log_rets[-min(self.vol_window, len(stock_log_rets)):]
                            rv = float(np.std(vol_rets)) * np.sqrt(252)
                            vols[sym] = rv if rv > 1e-6 else 1e-6

                        if len(scores) < 5:
                            if _TLT in live:
                                target[_TLT] = self.exposure
                        else:
                            k = min(self.top_k, len(scores))
                            ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
                            # Inverse-vol weighting
                            inv_vols = {sym: 1.0 / vols.get(sym, 1.0) for sym in ranked}
                            total_iv = sum(inv_vols.values())
                            for sym in ranked:
                                if sym in live:
                                    target[sym] = self.exposure * inv_vols[sym] / total_iv

        orders: list[Order] = []
        # Liquidate positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target
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
    return sp500_tickers() + [_TLT, _IEF, _SPY, _TYX, _TNX]


NAME = "idio_momentum_longend_slope"
HYPOTHESIS = (
    "Idiosyncratic SP500 momentum (63d return minus beta-adjusted SPY return) "
    "gated by long-end yield curve slope (TYX-TNX > 200d MA = risk-on top-15 stocks; "
    "TYX-TNX < 200d MA = SPY+IEF blend); SPY 200d bear gate to TLT; "
    "inverse-vol weighted; biweekly rebalance"
)

UNIVERSE = _universe

STRATEGY = IdioMomentumLongEndSlope()
