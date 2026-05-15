"""RSP equal-weight SP500 with portfolio vol-targeting — gen_10 sonnet-7

Hypothesis: Hold RSP (equal-weight SP500) when RSP is above its 200d SMA,
with exposure scaled to target a 12% annualized volatility (clip 50-97%).
When RSP below 200d SMA, hold TLT. Rebalance every 5 bars.

Rationale:
  - RSP is the equal-weight SP500 — fundamentally different from SPY.
    RSP outperforms during broad market rallies; SPY during mega-cap driven
    rallies. RSP IS exposure naturally tilts toward small/mid SP500.
  - Vol-targeting on RSP (not SPY) creates a different return path than
    gen9_sp500_voltarget_skipmon (which vol-targets a momentum STOCK portfolio,
    not RSP itself).
  - RSP as single ETF avoids the stock-selection turnover and corr issues
    of cross-sectional strategies.
  - The vol-targeting mechanism scales RSP exposure inversely to RSP's
    realized vol — auto-deleverages during volatile periods without
    being cal-year biased. Similar mechanism to gen9 vol-target but on
    equal-weight ETF.
  - This creates a "democratized market exposure with risk management" vs
    "concentrated momentum picks" — fundamentally different portfolio.
  - RSP covers IS (2003). TLT covers IS. SPY covers IS for trend gate.

Design:
  - Vol target: 12% annualized (0.12 / sqrt(252) daily).
  - 30-day realized vol used for exposure calculation.
  - Exposure clipped to [50%, 97%].
  - RSP 200d SMA gate: below -> TLT defensive.
  - Rebalance every 5 bars (weekly).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
TREND_WINDOW = 200        # RSP trend gate
VOL_WINDOW = 30           # realized vol for scaling
VOL_TARGET = 0.12         # 12% annualized
MIN_EXPOSURE = 0.50
MAX_EXPOSURE = 0.97


def _universe() -> list[str]:
    return ["RSP", "TLT", "SPY"]


UNIVERSE = _universe


class RspVoltarget(Strategy):
    """RSP equal-weight SP500 with 12% annualized vol-targeting; RSP 200d SMA
    gate to TLT defensive; weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        trend_window: int = TREND_WINDOW,
        vol_window: int = VOL_WINDOW,
        vol_target: float = VOL_TARGET,
        min_exposure: float = MIN_EXPOSURE,
        max_exposure: float = MAX_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            trend_window=trend_window,
            vol_window=vol_window,
            vol_target=vol_target,
            min_exposure=min_exposure,
            max_exposure=max_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.trend_window = int(trend_window)
        self.vol_window = int(vol_window)
        self.vol_target = float(vol_target)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.vol_window) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if p > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        # RSP 200d SMA trend gate
        rsp_hist = ctx.history("RSP")
        if len(rsp_hist) < self.trend_window + 5:
            return []
        rsp_close = rsp_hist["close"].dropna()
        if len(rsp_close) < self.trend_window:
            return []
        rsp_sma = float(rsp_close.iloc[-self.trend_window:].mean())
        rsp_bull = float(rsp_close.iloc[-1]) > rsp_sma

        target: dict[str, float] = {}

        if not rsp_bull:
            if "TLT" in live:
                target["TLT"] = self.max_exposure
        else:
            # Vol-targeting: scale RSP exposure to hit vol_target
            tail = rsp_close.values[-(self.vol_window + 1):]
            if len(tail) < self.vol_window + 1:
                return []
            logr = np.log(tail[1:] / tail[:-1])
            daily_vol = float(np.std(logr))
            if daily_vol <= 1e-8 or not np.isfinite(daily_vol):
                exposure = self.max_exposure
            else:
                ann_vol = daily_vol * np.sqrt(252.0)
                raw_exposure = self.vol_target / ann_vol
                exposure = float(np.clip(raw_exposure, self.min_exposure, self.max_exposure))

            if "RSP" in live:
                target["RSP"] = exposure

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


NAME = "rsp_voltarget"
HYPOTHESIS = (
    "RSP equal-weight SP500 with portfolio vol-targeting: exposure = clip(12pct_vol_target / "
    "30d_RSP_realized_vol, 50%, 97%); RSP 200d SMA gate -> TLT defensive; weekly rebalance — "
    "vol-targeting on equal-weight ETF vs gen9 vol-target on momentum stock portfolio"
)

STRATEGY = RspVoltarget()
