"""SP500 relative-strength-to-SPY selection with credit+trend dual gate.

Hypothesis: Select SP500 stocks that have both positive absolute momentum AND
are outperforming SPY on a relative basis. This dual screen (absolute + relative)
filters out stocks that look good in isolation but are laggards when the market
is rising. Inverse-vol weighting reduces concentration risk.

Signal:
  - Compute each stock's 63d relative return vs SPY (alpha to market)
  - Select top-15 stocks with BOTH positive relative alpha AND positive
    absolute 63d return
  - Dual gate: JNK above 30d SMA (credit ok) AND SPY above 200d SMA (bull)
  - Defensive: TLT when gate fails
  - Biweekly rebalance (10 bars) for more trades

Distinct from existing strategies:
  - Uses relative-to-SPY alpha for stock selection (not absolute momentum alone)
  - Screens for both positive alpha AND positive absolute return
  - JNK credit gate + SPY trend gate, but stock selection is entirely different
    from nearhi_momentum_quality (no near-52w-high filter)
  - Different from all pure momentum (63d, 126d, or composite) strategies
    because relative return to benchmark is the primary selection criterion
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10       # biweekly
TOP_K = 15
VOL_WINDOW = 20
TREND_WINDOW = 200
JNK_MA = 30
MOMENTUM_WINDOW = 63
EXPOSURE = 0.97

_SPY = "SPY"
_JNK = "JNK"


class SP500RSSpyGateInvVol(Strategy):
    """SP500 relative-strength selection (alpha-to-SPY) with JNK+SPY dual gate.

    Selects stocks with positive relative alpha AND positive absolute return
    over 63d. Inverse-vol sized. TLT defensive. Biweekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        top_k: int = TOP_K,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        jnk_ma: int = JNK_MA,
        momentum_window: int = MOMENTUM_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            top_k=top_k,
            vol_window=vol_window,
            trend_window=trend_window,
            jnk_ma=jnk_ma,
            momentum_window=momentum_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.top_k = int(top_k)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.jnk_ma = int(jnk_ma)
        self.momentum_window = int(momentum_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.momentum_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # SPY 200d SMA gate
        spy_bull = False
        spy_ret = None
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_bull = float(spy_close.iloc[-1]) > spy_sma
                # Also compute SPY momentum_window return for relative comparison
                if len(spy_close) >= self.momentum_window + 1:
                    spy_ret = float(spy_close.iloc[-1] / spy_close.iloc[-(self.momentum_window + 1)] - 1.0)
        except Exception:
            pass

        # JNK 30d SMA credit gate
        credit_ok = False
        try:
            jnk_hist = ctx.history(_JNK)
            if jnk_hist is not None and len(jnk_hist) >= self.jnk_ma:
                jnk_close = jnk_hist["close"].dropna()
                if len(jnk_close) >= self.jnk_ma:
                    jnk_sma = float(jnk_close.iloc[-self.jnk_ma:].mean())
                    credit_ok = float(jnk_close.iloc[-1]) > jnk_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull or not credit_ok or spy_ret is None:
            # Defensive: TLT
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # Risk-on: select top-K by relative alpha to SPY
            need = self.momentum_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.momentum_window:
                if "TLT" in live:
                    target["TLT"] = self.exposure
            else:
                alphas: dict[str, float] = {}
                inv_vols: dict[str, float] = {}

                for sym in prices.columns:
                    col = prices[sym].dropna()
                    if len(col) < self.momentum_window + 1:
                        continue

                    p_end = float(col.iloc[-1])
                    p_start = float(col.iloc[-(self.momentum_window + 1)])
                    if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
                        continue
                    abs_ret = (p_end / p_start) - 1.0
                    if not np.isfinite(abs_ret):
                        continue

                    # Require positive absolute return
                    if abs_ret <= 0:
                        continue

                    # Relative return (alpha to SPY)
                    rel_ret = abs_ret - spy_ret
                    if not np.isfinite(rel_ret):
                        continue

                    # Require positive relative return (outperforming SPY)
                    if rel_ret <= 0:
                        continue

                    # Inverse-vol weight
                    tail = col.iloc[-self.vol_window - 1:]
                    if len(tail) < self.vol_window + 1:
                        continue
                    logr = np.log(tail.values[1:] / tail.values[:-1])
                    rv = float(np.std(logr))
                    if rv <= 1e-6 or not np.isfinite(rv):
                        continue

                    alphas[sym] = rel_ret
                    inv_vols[sym] = 1.0 / rv

                if len(alphas) < 5:
                    # Fallback to TLT
                    if "TLT" in live:
                        target["TLT"] = self.exposure
                else:
                    k = min(self.top_k, len(alphas))
                    ranked = sorted(alphas, key=alphas.__getitem__, reverse=True)[:k]
                    iv_sum = sum(inv_vols[s] for s in ranked)
                    if iv_sum <= 0:
                        if "TLT" in live:
                            target["TLT"] = self.exposure
                    else:
                        for sym in ranked:
                            target[sym] = self.exposure * inv_vols[sym] / iv_sum

        # Build orders
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


NAME = "sp500_rs_spygate_invvol"
HYPOTHESIS = (
    "SP500 relative-strength-to-SPY selection: rank by 63d alpha vs SPY, hold top-15 with "
    "positive relative AND positive absolute return; JNK 30d SMA AND SPY 200d SMA dual gate; "
    "inverse-vol weighted; TLT defensive; biweekly rebalance; outperforming-SPY filter "
    "distinct from pure absolute momentum approaches"
)


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    return sp500_tickers() + ["TLT", "SPY", "JNK"]


UNIVERSE = _universe

STRATEGY = SP500RSSpyGateInvVol()
