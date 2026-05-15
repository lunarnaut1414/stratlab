"""gen_9 sonnet-3 — GDX/GLD Inflation Regime Stock-Selection Switcher

Hypothesis: Use GDX (gold miners ETF) vs GLD (gold bullion ETF) 42d return spread
as an inflation-expectations regime signal.

When miners lead bullion (GDX outperforms GLD on 42d return), it signals:
  - Real-asset inflation expectations rising (miners leverage gold price AND benefit from
    rising commodity costs, operational leverage, equity-market risk-on)
  - Macro: inflation regime favoring energy/materials/industrials
  -> Hold GLD 40% + top-5 XLE+XLB+XLI stocks by 63d momentum 57%
     (inflation-beneficiary stocks + physical gold hedge)

When gold bullion leads (GLD outperforms GDX), it signals:
  - Flight-to-safety, deflation, or gold as pure safe-haven (not equity-inflation)
  - Macro: normal or deflation/recession regime
  -> Hold top-15 SP500 stocks by 63d momentum 97%

SPY 200d SMA outer bear gate: both modes -> TLT 97%

Rationale: GDX/GLD spread captures the equity-vs-commodity split within gold, which
is a unique inflation-expectation proxy not correlated to VIX (fear), JNK (credit),
TNX/TYX (yield), or UUP (dollar). This is a fundamentally different economic signal.

No prior strategy in any round has used GDX/GLD as an inflation-regime switcher
that routes to DIFFERENT stock-selection baskets.
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

REBALANCE_EVERY = 10      # biweekly
MOM_WINDOW = 63           # momentum for SP500 and sector stocks
SPREAD_WINDOW = 42        # GDX vs GLD comparison window
TREND_WINDOW = 200        # SPY 200d SMA bear gate
TOP_K_SP500 = 15          # SP500 stocks in deflation/normal mode
TOP_K_INFL = 5            # energy/materials/industrials stocks in inflation mode
GLD_WEIGHT = 0.40         # GLD allocation in inflation mode
SECTOR_WEIGHT = 0.57      # sector stocks allocation in inflation mode
EXPOSURE = 0.97
_SPY = "SPY"
_TLT = "TLT"
_GLD = "GLD"
_GDX = "GDX"

# Inflation-regime sector stocks: energy, materials, industrials
INFLATION_TICKERS = [
    "XOM", "CVX", "SLB", "HAL", "COP", "EOG", "OXY",   # XLE / energy
    "LIN", "APD", "ECL", "FCX", "NEM", "NUE", "VMC",   # XLB / materials
    "CAT", "DE", "EMR", "HON", "GE", "MMM", "UNP",      # XLI / industrials
]


class GdxGldInflationRegime(Strategy):
    """GDX/GLD 42d spread as inflation-regime signal switching between SP500 momentum
    and inflation-beneficiary stock selection (energy+materials+industrials + GLD)."""

    def __init__(
        self,
        rebalance_every: int = REBALANCE_EVERY,
        mom_window: int = MOM_WINDOW,
        spread_window: int = SPREAD_WINDOW,
        trend_window: int = TREND_WINDOW,
        top_k_sp500: int = TOP_K_SP500,
        top_k_infl: int = TOP_K_INFL,
        gld_weight: float = GLD_WEIGHT,
        exposure: float = EXPOSURE,
    ) -> None:
        super().__init__(
            rebalance_every=rebalance_every,
            mom_window=mom_window,
            spread_window=spread_window,
            trend_window=trend_window,
            top_k_sp500=top_k_sp500,
            top_k_infl=top_k_infl,
            gld_weight=gld_weight,
            exposure=exposure,
        )
        self.rebalance_every = int(rebalance_every)
        self.mom_window = int(mom_window)
        self.spread_window = int(spread_window)
        self.trend_window = int(trend_window)
        self.top_k_sp500 = int(top_k_sp500)
        self.top_k_infl = int(top_k_infl)
        self.gld_weight = float(gld_weight)
        self.exposure = float(exposure)

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(self.trend_window, self.mom_window, self.spread_window) + 10
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

        # --- SPY 200d SMA outer bear gate ---
        spy_bull = True
        try:
            spy_hist = ctx.history(_SPY)
            if spy_hist is not None and len(spy_hist) >= self.trend_window + 2:
                spy_close = spy_hist["close"].dropna()
                if len(spy_close) >= self.trend_window:
                    spy_sma = float(spy_close.iloc[-self.trend_window:].mean())
                    spy_now = float(spy_close.iloc[-1])
                    spy_bull = spy_now > spy_sma
        except Exception:
            pass

        target: dict[str, float] = {}

        if not spy_bull:
            # Bear market — defensive TLT
            if _TLT in live:
                target[_TLT] = self.exposure
        else:
            # Determine inflation regime via GDX/GLD 42d return spread
            inflation_regime = False
            try:
                gdx_hist = ctx.history(_GDX)
                gld_hist = ctx.history(_GLD)
                if (gdx_hist is not None and gld_hist is not None and
                        len(gdx_hist) >= self.spread_window + 2 and
                        len(gld_hist) >= self.spread_window + 2):
                    gdx_close = gdx_hist["close"].dropna()
                    gld_close = gld_hist["close"].dropna()
                    if len(gdx_close) >= self.spread_window and len(gld_close) >= self.spread_window:
                        gdx_ret = float(gdx_close.iloc[-1] / gdx_close.iloc[-self.spread_window] - 1.0)
                        gld_ret = float(gld_close.iloc[-1] / gld_close.iloc[-self.spread_window] - 1.0)
                        if np.isfinite(gdx_ret) and np.isfinite(gld_ret):
                            inflation_regime = gdx_ret > gld_ret
            except Exception:
                pass

            if inflation_regime:
                # GDX outperforms GLD: inflation regime
                # Hold GLD 40% + top-5 energy/materials/industrials stocks by 63d momentum 57%
                if _GLD in live:
                    target[_GLD] = self.gld_weight * self.exposure

                # Score inflation-beneficiary stocks
                need = self.mom_window + 5
                prices = ctx.closes_window(need)
                infl_scores: dict[str, float] = {}
                for sym in INFLATION_TICKERS:
                    if sym not in prices.columns:
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                    if np.isfinite(ret) and sym in live:
                        infl_scores[sym] = ret

                if infl_scores:
                    ranked = sorted(infl_scores, key=infl_scores.__getitem__, reverse=True)
                    top_infl = ranked[:self.top_k_infl]
                    sector_alloc = (1.0 - self.gld_weight) * self.exposure
                    per_stock = sector_alloc / len(top_infl)
                    for sym in top_infl:
                        if sym in live:
                            target[sym] = per_stock
                else:
                    # No sector stocks available — add more GLD
                    if _GLD in live:
                        target[_GLD] = self.exposure
            else:
                # GLD leads or flat: normal/deflation regime — top-K SP500 momentum
                need = self.mom_window + 5
                prices = ctx.closes_window(need)
                scores: dict[str, float] = {}
                for sym in prices.columns:
                    if sym in (_SPY, _TLT, _GLD, _GDX):
                        continue
                    col = prices[sym].dropna()
                    if len(col) < self.mom_window:
                        continue
                    ret = float(col.iloc[-1] / col.iloc[-self.mom_window] - 1.0)
                    if np.isfinite(ret) and sym in live:
                        scores[sym] = ret

                if len(scores) < 5:
                    if _SPY in live:
                        target[_SPY] = self.exposure
                else:
                    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
                    top_stocks = ranked[:self.top_k_sp500]
                    per_weight = self.exposure / len(top_stocks)
                    for sym in top_stocks:
                        if sym in live:
                            target[sym] = per_weight

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


def _universe() -> list[str]:
    from stratlab.data.universe import sp500_tickers
    base = sp500_tickers()
    extras = [_TLT, _SPY, _GLD, _GDX] + INFLATION_TICKERS
    for t in extras:
        if t not in base:
            base.append(t)
    return base


UNIVERSE = _universe

NAME = "gdx_gld_inflation_regime"
HYPOTHESIS = (
    "GDX vs GLD 42d return spread as gold-miner inflation regime signal: "
    "when GDX outperforms GLD (miners leading = inflation expectations rising) "
    "hold GLD 40% + top-5 XLE+XLB+XLI stocks by 63d momentum 57%; "
    "when GLD leads or flat (deflation/safe-haven regime) hold top-15 SP500 by 63d momentum 97%; "
    "SPY 200d SMA bear gate to TLT; biweekly rebalance; "
    "inflation-regime stock-selection switching absent from leaderboard"
)

STRATEGY = GdxGldInflationRegime()
