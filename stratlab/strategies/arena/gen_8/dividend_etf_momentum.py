"""Dividend-quality ETF momentum rotation (gen_8, sonnet-1).

Hypothesis:
    Rotate among dividend-quality ETFs (VIG, DVY, SCHD) by 63-day return.
    These ETFs filter for dividend growth and income quality — a factor
    angle orthogonal to momentum (MTUM), low-vol (USMV), and growth (QQQ)
    rotations already on the leaderboard.

    * Hold top-2 dividend ETFs by 63d return when winner has positive 63d return
    * If all 3 are negative -> fallback to IEF (mid-duration bonds)
    * SPY bear (SPY below 200d SMA) -> hold TLT 97% (defensive override)
    * Monthly rebalance (every 21 bars)

    TNX direction (^TNX 20d MA vs 60d MA): if TNX rising (rising rates),
    add a 15% tilt toward DVY (higher-yield, shorter-duration income) and
    reduce VIG (dividend-growth, more rate-sensitive duration).

    Dividend-quality ETF rotation is absent from all prior gen_5 through
    gen_7 strategies. SCHD/VIG/DVY represent quality, income, and
    high-yield-income factor tilts not represented on leaderboard.
"""
from __future__ import annotations

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# --- Parameters ---
MOMENTUM_WINDOW = 63        # 3-month return for ETF ranking
TREND_WINDOW = 200          # SPY 200d SMA for bear gate
TNX_FAST = 20               # TNX fast MA for rate direction
TNX_SLOW = 60               # TNX slow MA for rate direction
REBALANCE_EVERY = 21        # monthly rebalance
EXPOSURE = 0.97

NAME = "dividend_etf_momentum"
HYPOTHESIS = (
    "Dividend growth ETF momentum rotation: rank VIG/DVY/SCHD by 63d return; "
    "hold top-2 equal-weight when winning ETF has positive 63d return; "
    "fallback to IEF when all negative; TNX direction tilt toward DVY in "
    "rising rates; SPY 200d SMA bear gate to TLT; monthly rebalance; "
    "dividend-quality ETF rotation absent from all prior rounds"
)


class DividendETFMomentum(Strategy):
    """Top-2 dividend ETFs by 3m return with TNX-tilt and SPY bear gate."""

    def __init__(
        self,
        momentum_window: int = MOMENTUM_WINDOW,
        trend_window: int = TREND_WINDOW,
        tnx_fast: int = TNX_FAST,
        tnx_slow: int = TNX_SLOW,
        rebalance_every: int = REBALANCE_EVERY,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            momentum_window=momentum_window,
            trend_window=trend_window,
            tnx_fast=tnx_fast,
            tnx_slow=tnx_slow,
            rebalance_every=rebalance_every,
            exposure=exposure,
        )
        self.momentum_window = int(momentum_window)
        self.trend_window = int(trend_window)
        self.tnx_fast = int(tnx_fast)
        self.tnx_slow = int(tnx_slow)
        self.rebalance_every = int(rebalance_every)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.momentum_window, self.trend_window, self.tnx_slow) + 5
        if ctx.idx < warmup:
            return []
        if ctx.idx % self.rebalance_every != 0:
            return []

        # --- SPY trend gate ---
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        spy_close = spy_hist["close"].dropna()
        if len(spy_close) < self.trend_window:
            return []
        spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
        spy_bull = float(spy_close.iloc[-1]) > spy_sma

        # --- TNX direction for rate-sensitivity tilt ---
        rates_rising = False
        try:
            tnx_hist = ctx.history("^TNX")
            tnx_close = tnx_hist["close"].dropna()
            if len(tnx_close) >= self.tnx_slow:
                tnx_fast_ma = float(tnx_close.iloc[-self.tnx_fast:].mean())
                tnx_slow_ma = float(tnx_close.iloc[-self.tnx_slow:].mean())
                rates_rising = tnx_fast_ma > tnx_slow_ma
        except KeyError:
            pass

        # --- Dividend ETF 63d returns ---
        div_etfs = ["VIG", "DVY", "SCHD"]
        scores: dict[str, float] = {}
        for sym in div_etfs:
            try:
                hist = ctx.history(sym)
            except KeyError:
                continue
            close = hist["close"].dropna()
            need = self.momentum_window + 2
            if len(close) < need:
                continue
            ret = float(close.iloc[-1]) / float(close.iloc[-(self.momentum_window + 1)]) - 1.0
            scores[sym] = ret

        if len(scores) == 0:
            return []

        # Bear override
        if not spy_bull:
            target = {"TLT": self.exposure}
        else:
            # Rank by 63d return
            ranked = sorted(scores, key=scores.__getitem__, reverse=True)
            # Take top-2 with positive momentum
            top2 = [s for s in ranked[:2] if scores.get(s, -1) > 0]

            if len(top2) == 0:
                # All negative -> IEF defensive
                target = {"IEF": self.exposure}
            else:
                # Equal weight top-2, apply TNX tilt if applicable
                base_weight = self.exposure / len(top2)
                target = {s: base_weight for s in top2}

                # TNX tilt: in rising rates, shift 15% from VIG to DVY if both present
                if rates_rising and "VIG" in target and "DVY" in target:
                    tilt = 0.15 * base_weight
                    target["DVY"] = target["DVY"] + tilt
                    target["VIG"] = max(0.0, target["VIG"] - tilt)

        # --- Execute ---
        closes_now = ctx.closes()
        if closes_now.empty:
            return []
        live = {s: float(p) for s, p in closes_now.items()}
        equity = ctx.portfolio_value(live)
        if equity <= 0:
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
            tgt_shares = int(equity * weight / price)
            cur = int(ctx.position(sym).size)
            delta = tgt_shares - cur
            if abs(delta) < 1:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            orders.append(Order(side=side, size=abs(delta), symbol=sym))

        return orders


STRATEGY = DividendETFMomentum()
UNIVERSE = ["SPY", "VIG", "DVY", "SCHD", "TLT", "IEF", "^TNX"]
