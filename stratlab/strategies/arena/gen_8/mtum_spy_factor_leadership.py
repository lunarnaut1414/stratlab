"""MTUM vs SPY Factor Leadership with Vol-Scaling — gen_8 sonnet-2

Hypothesis: Use the relative performance of MTUM (iShares MSCI USA Momentum
Factor ETF) vs SPY over 63 days to determine growth regime. When MTUM leads
SPY (momentum factor in favor), hold MTUM at vol-target exposure. When SPY
leads MTUM (broader market preferred over momentum factor), hold SPY at
vol-target. TLT when SPY below 200d SMA (bear). Weekly rebalance.

Rationale:
- MTUM vs SPY 63d spread is a factor-rotation signal (when momentum factor
  outperforms, stick with the factor; when it underperforms, rotate to the index)
- Vol-targeting scales exposure by inverse realized vol to maintain ~12% annual
  vol in the portfolio, providing automatic deleveraging in stress
- Routes exposure through factor ETFs (MTUM, SPY) not individual stocks —
  structurally distinct from SP500 cross-sectional strategies
- Distinct from curve/credit/VIX regime strategies in the leaderboard

Signal: MTUM 63d return vs SPY 63d return (factor momentum spread)
Sizing: Vol-target 12% annual using 21d realized vol, cap at 97%
Defensive: TLT when SPY < 200d SMA
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
MOM_WINDOW = 63           # 3 months for factor leadership
VOL_WINDOW = 21           # 21d for vol-target sizing
TREND_WINDOW = 200        # SPY 200d SMA gate
VOL_TARGET = 0.12         # 12% annual vol target
MAX_EXPOSURE = 0.97
MIN_EXPOSURE = 0.30       # don't go below 30% equity
ANNUAL_DAYS = 252.0
_MTUM = "MTUM"
_SPY = "SPY"
_TLT = "TLT"


class MTUMSPYFactorLeadership(Strategy):
    """MTUM vs SPY factor leadership rotation with vol-targeting."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        vol_window: int = VOL_WINDOW,
        trend_window: int = TREND_WINDOW,
        vol_target: float = VOL_TARGET,
        max_exposure: float = MAX_EXPOSURE,
        min_exposure: float = MIN_EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            vol_window=vol_window,
            trend_window=trend_window,
            vol_target=vol_target,
            max_exposure=max_exposure,
            min_exposure=min_exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.vol_window = int(vol_window)
        self.trend_window = int(trend_window)
        self.vol_target = float(vol_target)
        self.max_exposure = float(max_exposure)
        self.min_exposure = float(min_exposure)

    def _realized_vol_ann(self, sym: str, ctx: BarContext) -> float | None:
        """Return annualized realized vol from 21d returns, or None if insufficient."""
        try:
            hist = ctx.history(sym)
        except KeyError:
            return None
        close = hist["close"].dropna()
        if len(close) < self.vol_window + 2:
            return None
        tail = close.iloc[-(self.vol_window + 1):]
        log_rets = np.log(tail.values[1:] / tail.values[:-1])
        daily_vol = float(np.std(log_rets))
        if daily_vol <= 1e-8:
            return None
        return daily_vol * np.sqrt(ANNUAL_DAYS)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.mom_window, self.trend_window) + 10
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # SPY 200d SMA trend gate
        try:
            spy_hist = ctx.history(_SPY)
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_now = float(spy_close.iloc[-1])
        bull = spy_now > spy_sma

        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items() if float(p) > 0}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
            return []

        target: dict[str, float] = {}

        if not bull:
            # Bear regime: TLT defensive
            if _TLT in live:
                target[_TLT] = self.max_exposure
        else:
            # Compute 63d returns for MTUM and SPY
            need = self.mom_window + 5
            prices = ctx.closes_window(need)
            if len(prices) < self.mom_window:
                return []

            mtum_ret = None
            if _MTUM in prices.columns:
                col = prices[_MTUM].dropna()
                if len(col) >= self.mom_window + 1:
                    mtum_ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)

            spy_ret = None
            if _SPY in prices.columns:
                col = prices[_SPY].dropna()
                if len(col) >= self.mom_window + 1:
                    spy_ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)

            # Choose equity vehicle
            if mtum_ret is not None and spy_ret is not None and mtum_ret > spy_ret:
                equity_sym = _MTUM
            else:
                equity_sym = _SPY

            if equity_sym not in live:
                equity_sym = _SPY

            # Vol-target sizing
            ann_vol = self._realized_vol_ann(equity_sym, ctx)
            if ann_vol is not None and ann_vol > 0:
                raw_size = self.vol_target / ann_vol
                exposure = float(np.clip(raw_size, self.min_exposure, self.max_exposure))
            else:
                exposure = self.max_exposure

            if equity_sym in live:
                target[equity_sym] = exposure
            # Remaining cash sits idle (no TLT in bull with spare cash)

        orders: list[Order] = []

        # Exit positions not in target
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


UNIVERSE = [_MTUM, _SPY, _TLT]

NAME = "mtum_spy_factor_leadership"
HYPOTHESIS = (
    "MTUM vs SPY factor leadership: when MTUM 63d return > SPY 63d return hold MTUM; "
    "else hold SPY; vol-target 12% annual via 21d realized vol (cap 97%); "
    "TLT defensive when SPY < 200d SMA; weekly rebalance; factor momentum rotation "
    "through ETF vehicles, not individual stock selection"
)

STRATEGY = MTUMSPYFactorLeadership()
