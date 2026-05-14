"""TLT-IEF Convexity Signal SP500 Momentum — gen_8 sonnet-5

Hypothesis: Use the TLT vs IEF 10-day return differential as a "long-end
convexity / duration stress" signal for SP500 stock momentum.

Signal construction:
  convexity_spread = TLT_10d_return - IEF_10d_return

When TLT outperforms IEF (spread > 0):
  Long-end bonds are rallying more than medium-duration (7-10yr). This is
  a duration-positive, growth-supportive regime. Hold top-15 SP500 stocks
  by 63d momentum above their individual 50d SMA, inverse-vol weighted.

When TLT underperforms IEF (spread < -0.5%):
  Long-end rates are surging relative to medium-term rates. This often
  signals a "bear steepening" that is hostile to high-duration growth stocks.
  Rotate to SPY 60% + IEF 37% (lower-duration equity + medium bonds).

SPY 200d SMA outer bear gate: when SPY below 200d SMA, rotate to TLT 97%.

Rationale:
  The TLT-IEF spread is a novel, granular signal capturing the SLOPE of
  the long end of the yield curve's daily momentum. It differs from:
  - TNX level or direction (absolute rate level vs relative performance)
  - JNK/credit spread (credit risk, not duration risk)
  - VIX (equity volatility, not bond volatility)
  It is closer to the "long-end curve" signal in opus1_longend_curve_mtum_rotation
  (which uses TYX-TNX spread) but operates on daily momentum, not level spread.

  During IS 2010-2018: TLT repeatedly outperformed IEF during the multiple
  QE rounds (2011, 2012, 2015) and flight-to-quality episodes. Underperformed
  during taper tantrum 2013 and rate hike cycle 2015-2018 (bear steepener).

Rebalance: weekly (5 bars) for sufficient trade count.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5         # weekly
MOMENTUM_WINDOW = 63        # ~3 months
STOCK_TREND_WINDOW = 50     # individual stock 50d SMA
TREND_WINDOW = 200          # SPY 200d SMA
SPREAD_WINDOW = 10          # TLT-IEF 10d return spread
SPREAD_THRESHOLD = -0.005   # -0.5%: TLT underperforms IEF by >0.5% = stress
TOP_K = 15
EXPOSURE = 0.97
VOL_WINDOW = 21
_SPY = "SPY"
_TLT = "TLT"
_IEF = "IEF"


class TltIefConvexitySP500Momentum(Strategy):
    """TLT-IEF 10d spread gates SP500 momentum vs SPY+IEF defensive."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        momentum_window: int = MOMENTUM_WINDOW,
        stock_trend_window: int = STOCK_TREND_WINDOW,
        trend_window: int = TREND_WINDOW,
        spread_window: int = SPREAD_WINDOW,
        spread_threshold: float = SPREAD_THRESHOLD,
        top_k: int = TOP_K,
        exposure: float = EXPOSURE,
        vol_window: int = VOL_WINDOW,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            momentum_window=momentum_window,
            stock_trend_window=stock_trend_window,
            trend_window=trend_window,
            spread_window=spread_window,
            spread_threshold=spread_threshold,
            top_k=top_k,
            exposure=exposure,
            vol_window=vol_window,
        )
        self.rebalance_every = int(rebalance_every)
        self.momentum_window = int(momentum_window)
        self.stock_trend_window = int(stock_trend_window)
        self.trend_window = int(trend_window)
        self.spread_window = int(spread_window)
        self.spread_threshold = float(spread_threshold)
        self.top_k = int(top_k)
        self.exposure = float(exposure)
        self.vol_window = int(vol_window)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window, self.stock_trend_window) + 10
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

        # --- SPY trend gate (outer bear) ---
        spy_hist = ctx.history(_SPY)
        if len(spy_hist) < self.trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        spy_bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear: TLT 97%
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute TLT-IEF spread (10d return)
            tlt_hist = ctx.history(_TLT)
            ief_hist = ctx.history(_IEF)
            long_end_stressed = False  # default: not stressed

            if (len(tlt_hist) >= self.spread_window + 5 and
                    len(ief_hist) >= self.spread_window + 5):
                tlt_cl = tlt_hist["close"].dropna()
                ief_cl = ief_hist["close"].dropna()
                if len(tlt_cl) >= self.spread_window + 1 and len(ief_cl) >= self.spread_window + 1:
                    tlt_ret = float(tlt_cl.iloc[-1] / tlt_cl.iloc[-self.spread_window - 1] - 1.0)
                    ief_ret = float(ief_cl.iloc[-1] / ief_cl.iloc[-self.spread_window - 1] - 1.0)
                    spread = tlt_ret - ief_ret
                    long_end_stressed = spread < self.spread_threshold

            if long_end_stressed:
                # Long-end rates surging: hold SPY 60% + IEF 37%
                for sym, w in [(_SPY, 0.60), (_IEF, 0.37)]:
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                # Normal regime: hold top-K SP500 momentum stocks
                prices = ctx.closes_window(max(self.momentum_window, self.stock_trend_window) + 10)
                if len(prices) < self.momentum_window:
                    return []

                # Compute momentum scores
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _IEF):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.momentum_window] - 1.0)
                    if not np.isfinite(ret):
                        continue
                    scores[sym] = ret

                if len(scores) < 5:
                    if _IEF in live:
                        target[_IEF] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)

                    # Filter by individual 50d SMA and compute inverse-vol weights
                    selected_scores: dict[str, float] = {}
                    for sym in ranked:
                        if len(selected_scores) >= self.top_k:
                            break
                        sh = ctx.history(sym)
                        if len(sh) < self.stock_trend_window:
                            continue
                        sc = sh["close"].dropna()
                        if len(sc) < self.stock_trend_window:
                            continue
                        sma = float(sc.iloc[-self.stock_trend_window:].mean())
                        price = live.get(sym, 0.0)
                        if price <= sma:
                            continue
                        selected_scores[sym] = scores[sym]

                    if not selected_scores:
                        if _IEF in live:
                            target[_IEF] = self.exposure
                    else:
                        # Inverse-vol weighting
                        inv_vols: dict[str, float] = {}
                        for sym in selected_scores:
                            sh = ctx.history(sym)
                            if len(sh) < self.vol_window + 2:
                                inv_vols[sym] = 1.0
                                continue
                            sc = sh["close"].dropna()
                            if len(sc) < self.vol_window + 1:
                                inv_vols[sym] = 1.0
                                continue
                            rets = sc.pct_change().dropna().iloc[-self.vol_window:]
                            vol = float(rets.std())
                            inv_vols[sym] = 1.0 / vol if vol > 1e-8 else 1.0

                        total_inv_vol = sum(inv_vols.values())
                        if total_inv_vol <= 0:
                            per_w = self.exposure / len(selected_scores)
                            for sym in selected_scores:
                                if sym in live:
                                    target[sym] = per_w
                        else:
                            for sym, iv in inv_vols.items():
                                if sym in live:
                                    target[sym] = (iv / total_inv_vol) * self.exposure

        # --- Execute ---
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
    return sp500_tickers() + [_SPY, _TLT, _IEF]


NAME = "tlt_ief_convexity_sp500_momentum"
HYPOTHESIS = (
    "SP500 momentum with TLT-IEF convexity signal: hold top-15 SP500 stocks by 63d return "
    "above individual 50d SMA when TLT 10d return is above IEF 10d return minus 0.5% threshold "
    "(long-end bonds outperforming = duration-supportive regime); hold SPY 60%+IEF 37% when "
    "TLT underperforms IEF by >0.5% (long-end rates surging); SPY 200d SMA outer gate to TLT; "
    "inverse-vol weighted; weekly rebalance"
)

UNIVERSE = _universe

STRATEGY = TltIefConvexitySP500Momentum()
