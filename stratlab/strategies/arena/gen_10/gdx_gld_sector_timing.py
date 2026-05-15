"""GDX-vs-GLD momentum gate for sector ETF timing — gen_10 sonnet-7

Hypothesis: GDX (gold miners ETF) outperforming IAU (physical gold) over a
42-day window signals risk-on with real-economy demand (miners need economic
activity to mine profitably). When miners lead gold: hold risk-on sectors
(XLK tech + XLE energy). When gold leads miners (flight to quality, physical
gold > mining equity): hold defensive sectors (XLU utilities + hold TLT).
SPY 200d outer bear gate to TLT.

Rationale:
  - GDX/IAU (miners vs physical gold) is a classic real-economy risk signal.
    When GDX outperforms IAU, it implies miners are worth more than their gold
    reserves — economic activity and energy costs justify the premium.
  - This is a cross-asset signal (commodity + equity mining) driving SECTOR
    rotation, not individual stock selection. Completely different mechanism
    from all SP500 cross-sectional momentum strategies on leaderboard.
  - Sector ETFs (XLK, XLE, XLU) cover full IS window (starts 1998).
  - IAU covers IS (starts 2005; confirmed). GDX covers IS (starts 2006).
  - TLT defensive, SPY 200d gate, full IS coverage.
  - Expected OOS behavior: GDX/IAU signal captures global risk appetite
    distinct from VIX-level, credit spreads, and yield curve signals already
    saturated in leaderboard.

Design:
  - Compute GDX 42d return vs IAU 42d return.
  - Risk-on (GDX > IAU): hold XLK (50%) + XLE (47%) sectors.
  - Risk-off (IAU > GDX): hold XLU (50%) + TLT (47%).
  - SPY 200d SMA bear: hold TLT 97%.
  - Rebalance every 5 bars (weekly, same as idio_momentum-derived ideas).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 5       # weekly
SIGNAL_WINDOW = 42        # GDX vs IAU 42d return comparison
SPY_TREND_WINDOW = 200    # outer bear gate
EXPOSURE = 0.97


def _universe() -> list[str]:
    return ["GDX", "IAU", "XLK", "XLE", "XLU", "TLT", "SPY"]


UNIVERSE = _universe


class GdxGldSectorTiming(Strategy):
    """GDX vs IAU 42d return as risk-on/off signal for sector ETF rotation;
    SPY 200d outer bear gate to TLT; weekly rebalance.
    """

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        signal_window: int = SIGNAL_WINDOW,
        spy_trend_window: int = SPY_TREND_WINDOW,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            signal_window=signal_window,
            spy_trend_window=spy_trend_window,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.signal_window = int(signal_window)
        self.spy_trend_window = int(spy_trend_window)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.signal_window, self.spy_trend_window) + 5
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

        # SPY 200d outer bear gate
        spy_hist = ctx.history("SPY")
        if len(spy_hist) < self.spy_trend_window + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.spy_trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.spy_trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        target: dict[str, float] = {}

        if not spy_bull:
            if "TLT" in live:
                target["TLT"] = self.exposure
        else:
            # GDX vs IAU 42d return signal
            gdx_hist = ctx.history("GDX")
            iau_hist = ctx.history("IAU")

            risk_on = False  # default to risk-off if data missing
            if len(gdx_hist) >= self.signal_window + 2 and len(iau_hist) >= self.signal_window + 2:
                gdx_close = gdx_hist["close"].dropna()
                iau_close = iau_hist["close"].dropna()
                if (len(gdx_close) >= self.signal_window + 1 and
                        len(iau_close) >= self.signal_window + 1):
                    gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-self.signal_window] - 1.0)
                    iau_ret = float(iau_close.iloc[-1] / iau_close.iloc[-self.signal_window] - 1.0)
                    if np.isfinite(gdx_ret) and np.isfinite(iau_ret):
                        risk_on = gdx_ret > iau_ret

            if risk_on:
                # Risk-on: XLK (tech) + XLE (energy)
                half = self.exposure / 2.0
                if "XLK" in live:
                    target["XLK"] = half
                if "XLE" in live:
                    target["XLE"] = half
                # if only one available, double it
                if "XLK" not in live and "XLE" in live:
                    target["XLE"] = self.exposure
                elif "XLE" not in live and "XLK" in live:
                    target["XLK"] = self.exposure
            else:
                # Risk-off: XLU (defensive utilities) + TLT
                half = self.exposure / 2.0
                if "XLU" in live:
                    target["XLU"] = half
                if "TLT" in live:
                    target["TLT"] = half
                if "XLU" not in live and "TLT" in live:
                    target["TLT"] = self.exposure
                elif "TLT" not in live and "XLU" in live:
                    target["XLU"] = self.exposure

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


NAME = "gdx_gld_sector_timing"
HYPOTHESIS = (
    "SP500 126d momentum with per-stock 21d return acceleration filter: exclude stocks where "
    "21d return < 0 but 126d return > 0 (momentum decelerating); hold top-15 remaining stocks "
    "above their 126d SMA; inverse-vol weighted; SPY 200d outer bear gate to TLT; biweekly "
    "rebalance — acceleration filter is orthogonal to RSI quality filter"
)

STRATEGY = GdxGldSectorTiming()
