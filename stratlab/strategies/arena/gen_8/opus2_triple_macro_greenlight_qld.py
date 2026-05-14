"""Triple Cross-Asset Macro Green-Light with QLD Leverage Sleeve — gen_8 opus-2 (gap_finder)

Hypothesis: Use a 3-asset cross-asset confirmation rule as a regime "green light"
that justifies a small leveraged equity tilt, instead of binary on/off regime.

Three 42d-return signals (computed on tradeable ETFs, no signal-only series needed):
  - SPY 42d return > 0  : equity trending up
  - TLT 42d return > 0  : long bonds positive (duration tailwind / Fed accommodation)
  - UUP 42d return < 0  : dollar weakening (risk-on for global liquidity)

Green-light count (0..3):
  - 3 green       : QLD 50% + SPY 47% (mild 2x leverage tilt; QLD = 2x QQQ)
  - 2 green       : SPY 97% (clear bull, no leverage)
  - 1 green       : SPY 60% + IEF 37% (mixed)
  - 0 green       : IEF 60% + SHY 37% (defensive cash-equivalents)

Why this fills a gap:
- No prior strategy uses bond-equity-FX TRIPLE confirmation. Existing combos
  use one or two of these (UUP gating, TLT/IEF gating, SPY trend) but never
  all three as independent green-light votes.
- No prior strategy uses a LEVERAGED ETF (QLD) as the risk-on sleeve — only
  gen5_leveraged_etf_momentum used SSO (2x SPY), with weight=0.6. QLD has
  more cache history (2006-06) than QQQ-3x (TQQQ 2010-02), and 2x is less
  fragile than 3x for daily compounding decay during a multi-year hold.
- UUP signal is negated (dollar WEAKNESS = green) which is structurally
  distinct from the prior uup_dollar_trend_sp500 which uses UUP < 63d SMA
  as a positional gate rather than a vote.

Biweekly rebalance.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10
LOOKBACK = 42
EXPOSURE = 0.97

_SPY = "SPY"
_QLD = "QLD"
_TLT = "TLT"
_UUP = "UUP"
_IEF = "IEF"
_SHY = "SHY"


class TripleMacroGreenlightQLD(Strategy):
    """3-signal cross-asset confirmation gating QLD leverage sleeve."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        lookback: int = LOOKBACK,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            lookback=lookback,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.lookback = int(lookback)
        self.exposure = float(exposure)

    def _ret(self, ctx: BarContext, sym: str) -> float | None:
        try:
            h = ctx.history(sym)
        except KeyError:
            return None
        if h is None:
            return None
        cl = h["close"].dropna()
        if len(cl) < self.lookback + 1:
            return None
        try:
            r = float(cl.iloc[-1] / cl.iloc[-self.lookback] - 1.0)
        except Exception:
            return None
        if not np.isfinite(r):
            return None
        return r

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = self.lookback + 10
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

        spy_r = self._ret(ctx, _SPY)
        tlt_r = self._ret(ctx, _TLT)
        uup_r = self._ret(ctx, _UUP)

        if spy_r is None or tlt_r is None or uup_r is None:
            # Defensive fallback if any signal unavailable
            target: dict[str, float] = {}
            if _IEF in live:
                target[_IEF] = self.exposure * 0.62
            if _SHY in live:
                target[_SHY] = self.exposure * 0.38
        else:
            green = 0
            if spy_r > 0:
                green += 1
            if tlt_r > 0:
                green += 1
            if uup_r < 0:
                green += 1

            target = {}
            if green == 3:
                # Full green-light: QLD leveraged sleeve + SPY core
                if _QLD in live:
                    target[_QLD] = self.exposure * 0.52
                if _SPY in live:
                    target[_SPY] = self.exposure * 0.48
            elif green == 2:
                # Mostly green: clean SPY
                if _SPY in live:
                    target[_SPY] = self.exposure
            elif green == 1:
                # Mixed: SPY + IEF
                if _SPY in live:
                    target[_SPY] = self.exposure * 0.62
                if _IEF in live:
                    target[_IEF] = self.exposure * 0.38
            else:
                # Defensive: IEF + SHY
                if _IEF in live:
                    target[_IEF] = self.exposure * 0.62
                if _SHY in live:
                    target[_SHY] = self.exposure * 0.38

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
    return [_SPY, _QLD, _TLT, _UUP, _IEF, _SHY]


NAME = "opus2_triple_macro_greenlight_qld"
HYPOTHESIS = (
    "Triple cross-asset green-light gating QLD leverage sleeve: count of (SPY 42d>0, "
    "TLT 42d>0, UUP 42d<0). 3-green hold QLD 50%+SPY 47% (mild 2x QQQ leverage); "
    "2-green hold SPY 97%; 1-green hold SPY 60%+IEF 37%; 0-green hold IEF 60%+SHY 37%. "
    "Biweekly rebalance. Novel: bond-equity-FX triple-confirmation justifies leveraged "
    "ETF sleeve — no prior leaderboard strategy uses cross-asset vote-count gating."
)

UNIVERSE = _universe

STRATEGY = TripleMacroGreenlightQLD()
