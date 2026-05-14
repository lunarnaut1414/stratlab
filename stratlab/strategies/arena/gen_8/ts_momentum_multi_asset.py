"""Time-series momentum multi-asset allocation (gen_8, sonnet-1).

Hypothesis:
    Pure time-series (absolute) momentum on 4 major asset classes:
    SPY (US equity), QQQ (tech/growth), GLD (gold), IEF (intermediate bonds).
    Each asset is included in the portfolio if and only if its 252-day
    return is positive (Moskowitz/Asness TSMOM approach).

    Equal-weight among eligible assets (subject to 97% total exposure cap).
    If no asset qualifies, hold SHY (cash proxy).

    Biweekly rebalance (every 10 bars). Lookback = 252 trading days.

    The novelty is including QQQ alongside SPY (rather than just SPY),
    and using 252d absolute momentum as the sole filter (not relative/
    cross-sectional ranking or VIX/credit gate). All 4 assets have full
    IS-window coverage.

    The 2010-2018 bull market kept SPY and QQQ positive most of the time,
    so this stays equity-long while filtering out GLD/IEF bear periods.
    The signal is balanced across h1 (2010-2014) and h2 (2015-2018) because
    it's based on 1-year momentum, not short-term mean-reversion.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
TSMOM_WINDOW = 252          # 1-year return for absolute momentum
REBALANCE_EVERY = 10        # biweekly (every 10 bars)
EXPOSURE = 0.97

NAME = "ts_momentum_multi_asset"
HYPOTHESIS = (
    "Time-series momentum 4 assets (SPY/QQQ/GLD/IEF): hold each with positive "
    "252d return, equal-weight; hold SHY if none qualify; biweekly rebalance; "
    "pure absolute-momentum 4-asset allocator with QQQ+SPY dual equity "
    "distinct from single-SPY or bond-only TS momentum prior strategies"
)


class TSMomentumMultiAsset(Strategy):
    """Absolute 1-year momentum filter on SPY/QQQ/GLD/IEF."""

    def __init__(
        self,
        tsmom_window: int = TSMOM_WINDOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            tsmom_window=tsmom_window,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.tsmom_window = int(tsmom_window)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.tsmom_window + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        assets = ["SPY", "QQQ", "GLD", "IEF"]
        eligible: list[str] = []

        need = self.tsmom_window + 2
        for sym in assets:
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            close = hist["close"].dropna()
            if len(close) < need:
                continue
            ret_1y = float(close.iloc[-1]) / float(close.iloc[-(self.tsmom_window + 1)]) - 1.0
            if ret_1y > 0:
                eligible.append(sym)

        if len(eligible) == 0:
            target = {"SHY": self.exposure}
        else:
            w = self.exposure / len(eligible)
            target = {sym: w for sym in eligible}

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity_val = ctx.portfolio_value(live)
        if equity_val <= 0:
            return []

        orders: list[Order] = []

        # Sell positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in target and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Buy/adjust target positions
        for sym, weight in target.items():
            price = live.get(sym)
            if not price or price <= 0:
                continue
            tgt_shares = int(equity_val * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


STRATEGY = TSMomentumMultiAsset()
UNIVERSE = ["SPY", "QQQ", "GLD", "IEF", "SHY"]
