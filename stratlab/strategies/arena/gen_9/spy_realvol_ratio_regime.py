"""SPY Realized-Vol Ratio Regime Allocator — gen_9 sonnet-7

Hypothesis: Compute SPY's 5-day realized volatility vs its 63-day moving
average. This self-normalizing ratio detects whether current vol is calm
(expansion) or stressed (fear spike) relative to recent history.

Regime logic:
  - 5d_vol < 0.75 × 63d_avg_vol (calm expansion): QQQ 97% (high-beta growth
    outperforms in low-vol expansions)
  - 5d_vol 0.75x to 1.5x 63d_avg (neutral/normal): SPY 97% (broad equity)
  - 5d_vol > 1.5x 63d_avg_vol (fear spike): GLD 50%+IEF 47% (safe-haven
    assets that rally in genuine stress — gold + intermediate bonds)
  - SPY below 200d SMA outer gate: TLT 97% (full bear defensive)

Weekly rebalance (5 bars). Universe: popular_etfs (simpler universe, fast).

Rationale:
- The 5d/63d vol RATIO is self-normalizing: it adapts to any vol regime
  (2010's low vol, 2011's European crisis spike, 2018's vol shock).
- Unlike VIX level (absolute threshold), this ratio avoids the threshold-drift
  problem where IS-calibrated absolute levels fail OOS.
- The gen7 memo showed "threshold form (absolute, not percentile-rank) was
  fragile" for the SKEW wildcard. Same principle applies to vol thresholds.
- GLD+IEF in the stress bucket: GLD outperforms TLT in stagflation/inflation
  stress; IEF provides bond duration without the tail risk of TLT in rising-rate
  stress environments.

Distinct from all leaderboard strategies:
- SPY realized-vol ratio was tried in gen_7 (ic_58a7bf43) but failed at 0.11
  IS Calmar because it used SPY/QQQ/TLT routing. This variant:
  (1) uses GLD+IEF in stress bucket (not TLT alone)
  (2) uses 5d/63d vol ratio (not 10d/63d)
  (3) uses 63d average vol (not realized vol standard deviation of ratios)

The key change: GLD+IEF stress bucket instead of TLT reduces duration risk
during stress episodes that have RISING LONG RATES (e.g. 2013 taper tantrum).
"""
from __future__ import annotations

import numpy as np

from stratlab.engine.broker import Order, OrderSide
from stratlab.engine.context import BarContext
from stratlab.strategies.base import Strategy

# ── Parameters ──────────────────────────────────────────────────────────────
SHORT_VOL = 5        # short-window realized vol
LONG_VOL = 63        # long-window to compute average vol
SPY_TREND = 200      # outer bear gate
REBALANCE_DAYS = 5   # weekly rebalance

CALM_THRESHOLD = 0.75   # below this: calm expansion
STRESS_THRESHOLD = 1.50  # above this: fear spike

# Regime holdings
CALM = [("QQQ", 0.97)]
NEUTRAL = [("SPY", 0.97)]
STRESS = [("GLD", 0.50), ("IEF", 0.47)]
BEAR = [("TLT", 0.97)]


class SpyRealVolRatioRegime(Strategy):
    """SPY realized vol ratio gating QQQ/SPY/GLD+IEF/TLT."""

    def on_bar(self, ctx: BarContext) -> list[Order]:
        warmup = max(LONG_VOL + SHORT_VOL + 5, SPY_TREND + 5)
        if ctx.idx < warmup:
            return []
        if ctx.idx % REBALANCE_DAYS != 0:
            return []

        closes = ctx.closes()
        if closes.empty:
            return []
        live = {s: float(closes[s]) for s in closes.index if closes[s] > 0}

        # SPY outer bear gate
        try:
            spy_hist = ctx.history("SPY")
        except KeyError:
            return []
        if len(spy_hist) < SPY_TREND + 5:
            return []
        spy_close = spy_hist["close"].dropna()
        spy_sma200 = float(spy_close.iloc[-SPY_TREND:].mean())
        spy_price = float(spy_close.iloc[-1])
        spy_bull = spy_price > spy_sma200

        if not spy_bull:
            regime_holdings = BEAR
        else:
            # Compute SPY realized vol ratio
            need = LONG_VOL + SHORT_VOL + 2
            if len(spy_close) < need:
                return []

            # Compute daily log returns for the last LONG_VOL + SHORT_VOL bars
            tail = spy_close.iloc[-(need):].values
            log_rets = np.log(tail[1:] / tail[:-1])  # LONG_VOL + SHORT_VOL - 1 returns

            # SHORT_VOL realized vol (annualized for scaling)
            short_rets = log_rets[-SHORT_VOL:]
            short_vol = float(np.std(short_rets)) * np.sqrt(252)

            # LONG_VOL average daily realized vol (rolling SHORT_VOL windows)
            long_vols = []
            for i in range(LONG_VOL):
                window_rets = log_rets[i:i + SHORT_VOL]
                if len(window_rets) < SHORT_VOL:
                    continue
                long_vols.append(float(np.std(window_rets)) * np.sqrt(252))
            if not long_vols:
                return []
            avg_long_vol = float(np.mean(long_vols))

            if avg_long_vol < 1e-6:
                return []

            vol_ratio = short_vol / avg_long_vol

            if vol_ratio < CALM_THRESHOLD:
                regime_holdings = CALM    # calm expansion: go QQQ
            elif vol_ratio > STRESS_THRESHOLD:
                regime_holdings = STRESS  # fear spike: GLD+IEF
            else:
                regime_holdings = NEUTRAL  # neutral: SPY

        targets = dict(regime_holdings)

        # Compute portfolio equity
        equity = ctx.cash
        for sym, pos in ctx.positions.items():
            price = live.get(sym, 0.0)
            if price > 0:
                equity += pos.size * price
        if equity <= 0:
            return []

        orders: list[Order] = []

        # Exit positions not in target
        for sym, pos in list(ctx.positions.items()):
            if sym not in targets and pos.size != 0:
                side = OrderSide.SELL if pos.size > 0 else OrderSide.BUY
                orders.append(Order(side=side, size=abs(pos.size), symbol=sym))

        # Adjust to target weights
        for sym, weight in targets.items():
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
    return ["SPY", "QQQ", "GLD", "IEF", "TLT"]


NAME = "gen9_spy_realvol_ratio_regime"
HYPOTHESIS = (
    "SPY 5d realized vol vs 63d avg vol ratio as self-normalizing regime gate: "
    "ratio < 0.75 (calm expansion) -> QQQ 97%; "
    "ratio 0.75-1.5 (neutral) -> SPY 97%; "
    "ratio > 1.5 (fear spike) -> GLD 50%+IEF 47%; "
    "SPY bear -> TLT 97%. Weekly rebalance."
)

UNIVERSE = _universe

STRATEGY = SpyRealVolRatioRegime()
