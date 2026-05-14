"""Gold/Bond Ratio Regime — gen_7 sonnet-7 (attempt 7)

Hypothesis: Use the GLD/TLT price ratio trend (20d vs 60d MA) as an inflation
vs deflation regime signal:

- GLD/TLT ratio rising (gold beating bonds): inflationary regime
  -> hold XLE (energy) 40% + GLD 30% + IWM 27%
- GLD/TLT ratio falling (bonds beating gold): deflationary/flight-to-safety
  -> hold TLT 60% + IEF 37%
- SPY 200d SMA gate: when SPY<200d override to TLT 97%

Rationale: The gold-to-bond ratio is a clean macro inflation barometer.
When gold outperforms bonds, inflation expectations are rising and cyclical/
commodity assets outperform. When bonds outperform gold, deflation risk is
elevated and long-duration bonds outperform.

This signal is entirely distinct from:
- JNK credit spread (credit risk, not inflation)
- VIX level (fear, not inflation)
- TNX direction (rate direction, not gold/bond spread)
- SP500 momentum (stock selection, not asset class rotation)

The output holdings (XLE+GLD+IWM in inflation; TLT+IEF in deflation) are
also novel vs existing leaderboard.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

UNIVERSE = ["GLD", "TLT", "IEF", "XLE", "IWM", "SPY"]

REBALANCE_EVERY = 5        # weekly
FAST_MA = 20               # fast MA on GLD/TLT ratio
SLOW_MA = 60               # slow MA on GLD/TLT ratio
TREND_WINDOW = 200
EXPOSURE = 0.97
_SPY = "SPY"
_GLD = "GLD"
_TLT = "TLT"
_IEF = "IEF"
_XLE = "XLE"
_IWM = "IWM"


class GoldBondRatioRegime(Strategy):
    """Inflation regime via GLD/TLT ratio MA crossover: inflationary (cyclical/GLD) vs
    deflationary (TLT+IEF) rotation.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        fast_ma: int = FAST_MA,
        slow_ma: int = SLOW_MA,
        trend_window: int = TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            fast_ma=fast_ma,
            slow_ma=slow_ma,
            trend_window=trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.fast_ma = int(fast_ma)
        self.slow_ma = int(slow_ma)
        self.trend_window = int(trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.slow_ma, self.trend_window) + 10
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

        # SPY 200d SMA gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            spy_hist = None

        bull = True
        if spy_hist is not None and len(spy_hist) >= self.trend_window:
            spy_close = spy_hist["close"].dropna()
            if len(spy_close) >= self.trend_window:
                spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                spy_now = float(spy_close.iloc[-1])
                bull = spy_now > spy_sma

        target: dict[str, float] = {}

        if not bull:
            # Bear market: full TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Compute GLD/TLT ratio MA crossover
            need = self.slow_ma + 5
            prices = ctx.closes_window(need)

            inflation_regime = True  # default
            if (len(prices) >= self.slow_ma and
                    _GLD in prices.columns and _TLT in prices.columns):
                gld = prices[_GLD].dropna()
                tlt = prices[_TLT].dropna()
                # Align lengths
                min_len = min(len(gld), len(tlt))
                if min_len >= self.slow_ma:
                    gld_tail = gld.iloc[-min_len:]
                    tlt_tail = tlt.iloc[-min_len:]
                    # Avoid division by zero
                    ratio = gld_tail.values / tlt_tail.values
                    if not np.any(np.isnan(ratio)) and not np.any(ratio <= 0):
                        fast_ma_val = float(ratio[-self.fast_ma:].mean())
                        slow_ma_val = float(ratio[-self.slow_ma:].mean())
                        inflation_regime = fast_ma_val > slow_ma_val

            if inflation_regime:
                # Inflationary: energy + gold + small caps
                weights = {_XLE: 0.40, _GLD: 0.30, _IWM: 0.27}
                for sym, w in weights.items():
                    if sym in live:
                        target[sym] = w * self.exposure
            else:
                # Deflationary: long-duration bonds
                tlt_w = 0.60
                ief_w = 0.37
                if _TLT in live:
                    target[_TLT] = tlt_w * self.exposure
                if _IEF in live:
                    target[_IEF] = ief_w * self.exposure

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


NAME = "gold_bond_ratio_regime"
HYPOTHESIS = (
    "GLD/TLT ratio MA crossover inflation regime: when GLD outperforming TLT (20d vs 60d MA) "
    "hold XLE 40%+GLD 30%+IWM 27% (inflation); else hold TLT 60%+IEF 37% (deflation); "
    "SPY 200d SMA bear gate to TLT; weekly rebalance; pure macro inflation signal orthogonal "
    "to credit/VIX/momentum signals on leaderboard"
)

STRATEGY = GoldBondRatioRegime()
